import os
import time
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Tuple

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

import logging

from transformers import (InputFeatures,
                          InputExample,
                          PreTrainedTokenizerBase)

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


import sys
sys.path.append("../")
sys.path.append('./biobert_re/')

from data_processor import glue_convert_examples_to_features, glue_output_modes, glue_processors

import utils
from ehr import HealthRecord
from annotations import Relation

logger = logging.getLogger(__name__)


@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command
    line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


# noinspection PyTypeChecker
class REDataset(Dataset):
    """
    A class representing a training dataset for Relation Extraction.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )

        label_list = self.processor.get_labels()

        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start)
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)

                logger.info("Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


class RETestDataset(Dataset):
    """
    A class representing a test Dataset for relation extraction.
    """

    def __init__(self, test_ehr, tokenizer, max_seq_len, label_list):

        self.re_text_list, self.relation_list = generate_re_test_file(test_ehr)

        if not self.re_text_list:
            self.features = []
        else:
            examples = []
            for (i, text) in enumerate(self.re_text_list):
                guid = "%s" % i
                examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=None))

            self.features = glue_convert_examples_to_features(examples, tokenizer,
                                                              max_length=max_seq_len,
                                                              label_list=label_list)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def replace_ent_label(text, ent_type, start_idx, end_idx):
    label = '@'+ent_type+'$'
    return text[:start_idx]+label+text[end_idx:]


def write_file(file, index, sentence, label, sep, is_test, is_label):
    if is_test and is_label:  # test_original - test with labels
        file.write('{}{}{}{}{}'.format(index, sep, sentence, sep, label))
    elif is_test and not is_label:  # test - test with no labels
        file.write('{}{}{}'.format(index, sep, sentence))
    else:  # train
        file.write('{}{}{}'.format(sentence, sep, label))
    file.write('\n')


def get_char_split_points(record, max_len):
    char_split_points = []

    split_points = record.get_split_points(max_len=max_len)
    for pt in split_points[:-1]:
        char_split_points.append(record.get_char_idx(pt)[1])

    if len(char_split_points) == 1:
        return char_split_points
    else:
        return char_split_points[1:]


def replace_entity_text(split_text, ent1, ent2, split_offset):
    # Remove split offset
    ent1_start = ent1.range[0] - split_offset
    ent1_end = ent1.range[1] - split_offset

    ent2_start = ent2.range[0] - split_offset
    ent2_end = ent2.range[1] - split_offset

    # If entity 1 is present before entity 2
    if ent1_end < ent2_end:
        # Insert entity 2 and then entity 1
        modified_text = replace_ent_label(split_text, ent2.name, ent2_start, ent2_end)
        modified_text = replace_ent_label(modified_text, ent1.name, ent1_start, ent1_end)

    # If entity 1 is present after entity 2
    else:
        # Insert entity 1 and then entity 2
        modified_text = replace_ent_label(split_text, ent1.name, ent1_start, ent1_end)
        modified_text = replace_ent_label(modified_text, ent2.name, ent2_start, ent2_end)

    return modified_text


def generate_re_input_files(ehr_records: List[HealthRecord], filename: str,
                            ade_records: List[Dict] = None, max_len: int = 128,
                            is_test=False, is_label=True, is_predict=False, sep: str = '\t'):

    random.seed(0)

    index = 0
    index_rel_label_map = []

    with open(filename, 'w') as file:
        # Write headers
        write_file(file, 'index', 'sentence', 'label', sep, is_test, is_label)

        # Preprocess EHR records
        for record in ehr_records:
            text = record.text
            entities = record.get_entities()

            if is_predict:
                true_relations = None
            else:
                true_relations = record.get_relations()

            # get character split points
            char_split_points = get_char_split_points(record, max_len)

            start = 0
            end = char_split_points[0]

            for i in range(len(char_split_points)):
                # Obtain only entities within the split text
                range_entities = {ent_id: ent for ent_id, ent in
                                  filter(lambda item: int(item[1][0]) >= start and int(item[1][1]) <= end,
                                         entities.items())}

                # Get all possible relations within the split text
                possible_relations = utils.map_entities(range_entities, true_relations)

                for rel, label in possible_relations:
                    if label == 0 and rel.name != "ADE-Drug":
                        if random.random() > 0.25:
                            continue

                    split_text = text[start:end]
                    split_offset = start

                    ent1 = rel.get_entities()[0]
                    ent2 = rel.get_entities()[1]

                    # Check if both entities are within split text
                    if ent1.range[0] >= start and ent1.range[1] < end and \
                            ent2.range[0] >= start and ent2.range[1] < end:

                        modified_text = replace_entity_text(split_text, ent1, ent2, split_offset)

                        # Replace un-required characters with space
                        final_text = modified_text.replace('\n', ' ').replace('\t', ' ')
                        write_file(file, index, final_text, label, sep, is_test, is_label)

                        if is_predict:
                            index_rel_label_map.append({'relation': rel})
                        else:
                            index_rel_label_map.append({'label': label, 'relation': rel})

                        index += 1

                start = end
                if i != len(char_split_points)-1:
                    end = char_split_points[i+1]
                else:
                    end = len(text)+1

        # Preprocess ADE records
        if ade_records is not None:
            for record in ade_records:
                entities = record['entities']
                true_relations = record['relations']
                possible_relations = utils.map_entities(entities, true_relations)

                for rel, label in possible_relations:

                    if label == 1 and random.random() > 0.5:
                        continue

                    new_tokens = record['tokens'].copy()

                    for ent in rel.get_entities():
                        ent_type = ent.name

                        start_tok = ent.range[0]
                        end_tok = ent.range[1]+1

                        for i in range(start_tok, end_tok):
                            new_tokens[i] = '@'+ent_type+'$'

                    """Remove consecutive repeating entities.
                    Eg. this is @ADE$ @ADE$ @ADE$ for @Drug$ @Drug$ -> this is @ADE$ for @Drug$"""
                    final_tokens = [new_tokens[i] for i in range(len(new_tokens))\
                                    if (i == 0) or new_tokens[i] != new_tokens[i-1]]

                    final_text = " ".join(final_tokens)

                    write_file(file, index, final_text, label, sep, is_test, is_label)
                    index_rel_label_map.append({'label': label, 'relation': rel})
                    index += 1

    filename, ext = filename.split('.')
    utils.save_pickle(filename+'_rel.pkl', index_rel_label_map)


def get_eval_results(answer_path, output_path):
    """
    Get evaluation metrics for predictions

    Parameters
    ------------
    answer_path : test.tsv file. Tab-separated.
                  One example per a line. True labels at the 3rd column.

    output_path : test_predictions.txt. Model generated predictions.
    """
    testdf = pd.read_csv(answer_path, sep="\t", index_col=0)
    preddf = pd.read_csv(output_path, sep="\t", header=None)

    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [int(v[1]) for v in pred[1:]]

    p, r, f, s = precision_recall_fscore_support(y_pred=pred_class, y_true=testdf["label"])
    results = dict()
    results["f1 score"] = f[1]
    results["recall"] = r[1]
    results["precision"] = p[1]
    results["specificity"] = r[0]
    return results


def generate_re_test_file(ehr_record: HealthRecord,
                          max_len: int = 128) -> Tuple[List[str], List[Relation]]:
    """
    Generates test file for Relation Extraction.

    Parameters
    -----------
    ehr_record : HealthRecord
        The EHR record with entities set.

    max_len : int
        The maximum length of sequence.

    Returns
    --------
    Tuple[List[str], List[Relation]]
        List of sequences with entity replaced by it's tag.
        And a list of relation objects representing relation in those sequences.
    """
    random.seed(0)

    re_text_list = []
    relation_list = []

    text = ehr_record.text
    entities = ehr_record.get_entities()
    if isinstance(entities, dict):
        entities = list(entities.values())

    # get character split points
    char_split_points = get_char_split_points(ehr_record, max_len)

    start = 0
    end = char_split_points[0]

    for i in range(len(char_split_points)):
        # Obtain only entities within the split text
        range_entities = [ent for ent in filter(lambda item: int(item[0]) >= start and int(item[1]) <= end,
                                                entities)]

        # Get all possible relations within the split text
        possible_relations = utils.map_entities(range_entities)

        for rel, label in possible_relations:
            split_text = text[start:end]
            split_offset = start

            ent1 = rel.get_entities()[0]
            ent2 = rel.get_entities()[1]

            # Check if both entities are within split text
            if ent1[0] >= start and ent1[1] < end and \
                    ent2[0] >= start and ent2[1] < end:

                modified_text = replace_entity_text(split_text, ent1, ent2, split_offset)

                # Replace un-required characters with space
                final_text = modified_text.replace('\n', ' ').replace('\t', ' ')

                re_text_list.append(final_text)
                relation_list.append(rel)

        start = end
        if i != len(char_split_points)-1:
            end = char_split_points[i+1]
        else:
            end = len(text)+1

    assert len(re_text_list) == len(relation_list)

    return re_text_list, relation_list

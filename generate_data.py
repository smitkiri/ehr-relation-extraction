import argparse

from utils import read_data, save_pickle, read_ade_data
from biobert_ner.utils_ner import generate_input_files
from biobert_re.utils_re import generate_re_input_files
from typing import List, Iterator, Dict
import warnings
import os
import re

labels = ['B-DRUG', 'I-DRUG', 'B-STR', 'I-STR', 'B-DUR', 'I-DUR',
          'B-ROU', 'I-ROU', 'B-FOR', 'I-FOR', 'B-ADE', 'I-ADE',
          'B-DOS', 'I-DOS', 'B-REA', 'I-REA', 'B-FRE', 'I-FRE', 'O']


def parse_arguments():
    """Parses program arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,
                        help="Task to be completed. 'NER', 'RE'. Default is 'NER'.",
                        default="NER")

    parser.add_argument("--input_dir", type=str,
                        help="Directory with txt and ann files. Default is 'data/'.",
                        default="data/")

    parser.add_argument("--ade_dir", type=str,
                        help="Directory with ADE corpus. Default is None.",
                        default=None)

    parser.add_argument("--target_dir", type=str,
                        help="Directory to save files. Default is 'dataset/'.",
                        default='dataset/')

    parser.add_argument("--max_seq_len", type=int,
                        help="Maximum sequence length. Default is 512.",
                        default=512)

    parser.add_argument("--dev_split", type=float,
                        help="Ratio of dev data. Default is 0.1",
                        default=0.1)

    parser.add_argument("--tokenizer", type=str,
                        help="The tokenizer to use. 'scispacy', 'scispacy_plus', 'biobert-base', 'biobert-large', 'default'.",
                        default="scispacy")

    parser.add_argument("--ext", type=str,
                        help="Extension of target file. Default is txt.",
                        default="txt")

    parser.add_argument("--sep", type=str,
                        help="Token-label separator. Default is a space.",
                        default=" ")

    arguments = parser.parse_args()
    return arguments


def default_tokenizer(sequence: str) -> List[str]:
    """A tokenizer that splits sequence by a whitespace."""
    words = re.split("\n| |\t", sequence)
    tokens = []
    for word in words:
        word = word.strip()

        if not word:
            continue

        tokens.append(word)

    return tokens


def scispacy_plus_tokenizer(sequence: str, scispacy_tok=None) -> Iterator[str]:
    """
    Runs the scispacy tokenizer and removes all tokens with
    just whitespace characters
    """
    if scispacy_tok is None:
        import en_ner_bc5cdr_md
        scispacy_tok = en_ner_bc5cdr_md.load().tokenizer

    scispacy_tokens = list(map(lambda x: str(x), scispacy_tok(sequence)))
    tokens = filter(lambda t: not (' ' in t or '\n' in t or '\t' in t), scispacy_tokens)

    return tokens


def ner_generator(files: Dict[str, tuple], args) -> None:
    """Generates files for NER"""
    # Generate train, dev, test files
    for filename, data in files.items():
        generate_input_files(ehr_records=data[0], ade_records=data[1],
                             filename=args.target_dir + filename + '.' + args.ext,
                             max_len=args.max_seq_len, sep=args.sep)
        save_pickle(args.target_dir + filename, {"EHR": data[0], "ADE": data[1]})

    # Generate labels file
    with open(args.target_dir + 'labels.txt', 'w') as file:
        output_labels = map(lambda x: x + '\n', labels)
        file.writelines(output_labels)

    filenames = [name for files in map(
        lambda x: [x + '.' + args.ext, x + '.pkl'],
        list(files.keys()))
                 for name in files]

    print("\nGenerating files successful. Files generated: ",
          ', '.join(filenames), ', labels.txt', sep='')


def re_generator(files: Dict[str, tuple], args):
    """Generates files for RE"""
    for filename, data in files.items():
        generate_re_input_files(ehr_records=data[0], ade_records=data[1],
                                filename=args.target_dir + filename + '.' + args.ext,
                                max_len=args.max_seq_len, sep=args.sep,
                                is_test=data[2], is_label=data[3])

    save_pickle(args.target_dir + 'train', {"EHR": files['train'][0], "ADE": files['train'][1]})
    save_pickle(args.target_dir + 'test',  {"EHR": files['test'][0],  "ADE": files['test'][1]})

    print("\nGenerating files successful. Files generated: ",
          'train.tsv,', 'dev.tsv,', 'test.tsv,',
          'test_labels.tsv,', 'train_rel.pkl,', 'test_rel.pkl,', 'test_labels_rel.pkl', sep=' ')


def main():
    args = parse_arguments()

    if args.target_dir[-1] != '/':
        args.target_dir += '/'

    if args.sep == "tab":
        args.sep = "\t"

    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    if args.tokenizer == "default":
        tokenizer = default_tokenizer
        is_bert_tokenizer = False

    elif args.tokenizer == "scispacy":
        import en_ner_bc5cdr_md
        tokenizer = en_ner_bc5cdr_md.load().tokenizer
        is_bert_tokenizer = False

    elif args.tokenizer == 'scispacy_plus':
        import en_ner_bc5cdr_md
        scispacy_tok = en_ner_bc5cdr_md.load().tokenizer
        scispacy_plus_tokenizer.__defaults__ = (scispacy_tok,)

        tokenizer = scispacy_plus_tokenizer
        is_bert_tokenizer = False

    elif args.tokenizer == 'biobert-large':
        from transformers import AutoTokenizer
        biobert = AutoTokenizer.from_pretrained(
            "dmis-lab/biobert-large-cased-v1.1")

        args.max_seq_len -= biobert.num_special_tokens_to_add()
        tokenizer = biobert.tokenize
        is_bert_tokenizer = True

    elif args.tokenizer == 'biobert-base':
        from transformers import AutoTokenizer
        biobert = AutoTokenizer.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.1")

        args.max_seq_len -= biobert.num_special_tokens_to_add()
        tokenizer = biobert.tokenize
        is_bert_tokenizer = True

    else:
        warnings.warn("Tokenizer named " + args.tokenizer + " not found."
                      "Using default tokenizer instead. Acceptable values"
                      "include 'scispacy', 'biobert-base', 'biobert-large',"
                      "and 'default'.")
        tokenizer = default_tokenizer
        is_bert_tokenizer = False

    print("\nReading data\n")
    train_dev, test = read_data(data_dir=args.input_dir,
                                tokenizer=tokenizer,
                                is_bert_tokenizer=is_bert_tokenizer,
                                verbose=1)

    if args.ade_dir is not None:
        ade_train_dev = read_ade_data(ade_data_dir=args.ade_dir, verbose=1)

        ade_dev_split_idx = int((1 - args.dev_split) * len(ade_train_dev))
        ade_train = ade_train_dev[:ade_dev_split_idx]
        ade_devel = ade_train_dev[ade_dev_split_idx:]

    else:
        ade_train_dev = None
        ade_train = None
        ade_devel = None

    print('\n')

    # Data is already shuffled, just split for dev set
    dev_split_idx = int((1 - args.dev_split) * len(train_dev))
    train = train_dev[:dev_split_idx]
    devel = train_dev[dev_split_idx:]

    # Data for NER
    if args.task.lower() == 'ner':
        files = {'train': (train, ade_train), 'train_dev': (train_dev, ade_train_dev),
                 'devel': (devel, ade_devel), 'test': (test, None)}

        ner_generator(files, args)

    # Data for RE
    elif args.task.lower() == 're':
        # {dataset_name: (ehr_data, ade_data, is_test, is_label)}
        files = {'train': (train, ade_train, False, True), 'dev': (devel, ade_devel, False, True),
                 'test': (test, None, True, False), 'test_labels': (test, None, True, True)}

        re_generator(files, args)


if __name__ == '__main__':
    main()

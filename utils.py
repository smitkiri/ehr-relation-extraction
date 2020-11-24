from typing import List, Tuple, Callable, Dict, Union
from annotations import Entity, Relation

import os
import sys
from pickle import dump, load
from IPython.core.display import display, HTML
from ehr import HealthRecord
import random
import json
from collections import defaultdict

TPL_HTML = """<span style = "background-color: {color}; border-radius: 5px;">&nbsp;{content}&nbsp;</span>"""

TPL_HTML_HOVER = """<span style = "background-color: {color}; border-radius: 5px;">&nbsp;{content}&nbsp;<span style = "background: {color}">{ent_type}</span></span>"""

COLORS = {"Drug": "#aa9cfc", "Strength": "#ff9561",
          "Form": "#7aecec", "Frequency": "#9cc9cc",
          "Route": "#ffeb80", "Dosage": "#bfe1d9",
          "Reason": "#e4e7d2", "ADE": "#ff8197",
          "Duration": "#97c4f5"}


# noinspection PyTypeChecker
def display_ehr(text: str,
                entities: Union[Dict[str, Entity], List[Entity]],
                return_html: bool = False) -> Union[None, str]:
    """
    Highlights EHR records with colors and displays
    them as HTML. Ideal for working with Jupyter Notebooks

    Parameters
    ----------
    text : str
        EHR record to render

    entities : Union[Dict[str, Entity], List[Entity]]
         A list of Entity objects

    return_html : bool
        Indicator for returning HTML or printing it.

    Returns
    -------
    Union[None, str]
        If return_html is true, returns html strings
        otherwise displays HTML.

    """
    if isinstance(entities, dict):
        entities = list(entities.values())

    # Sort entity by starting range
    entities.sort(key=lambda x: x.range[0])

    # Final text to render
    render_text = ""
    start_idx = 0

    if not return_html:
        # Display legend
        for ent, col in COLORS.items():
            render_text += TPL_HTML.format(content=ent, color=col)
            render_text += "&nbsp" * 5

        render_text += '\n'
        render_text += '--' * 50
        render_text += "\n\n"

    # Replace each character range with HTML span template
    for ent in entities:
        if start_idx > ent.range[0]:
            continue

        render_text += text[start_idx:ent.range[0]]

        if return_html:
            render_text += TPL_HTML_HOVER.format(
                content=text[ent.range[0]:ent.range[1]],
                color=COLORS[ent.name],
                ent_type=ent.name)
        else:
            render_text += TPL_HTML.format(
                content=text[ent.range[0]:ent.range[1]],
                color=COLORS[ent.name])

        start_idx = ent.range[1]

    render_text += text[start_idx:]
    render_text = render_text.replace("\n", "<br>")

    if return_html:
        return render_text
    else:
        # Render HTML
        display(HTML(render_text))


def read_data(data_dir: str = 'data/',
              train_ratio: int = 0.8,
              tokenizer: Callable[[str], List[str]] = None,
              verbose: int = 0) -> Tuple[List[HealthRecord], List[HealthRecord]]:
    """
    Reads train and test data

    Parameters
    ----------
    data_dir : str, optional
        Directory where the data is located. The default is 'data/'.

    train_ratio : int, optional
        Percentage split of train data. The default is 0.8.

    tokenizer : Callable[[str], List[str]], optional
        The tokenizer function to use.. The default is None.

    verbose : int, optional
        1 to print reading progress, 0 otherwise. The default is 0.

    Returns
    -------
    Tuple[List[HealthRecord], List[HealthRecord]]
        Train data, Test data.

    """
    # Get all the IDs of data
    file_ids = sorted(list(set(['.'.join(fname.split('.')[:-1]) \
                                for fname in os.listdir(data_dir) \
                                if not fname.startswith('.')])))

    # Splitting IDs into random training and test data
    random.seed(0)
    random.shuffle(file_ids)

    split_idx = int(train_ratio * len(file_ids))
    train_ids = file_ids[:split_idx]
    test_ids = file_ids[split_idx:]

    if verbose == 1:
        print("Train data:")

    train_data = []
    for idx, fid in enumerate(train_ids):
        record = HealthRecord(fid, text_path=data_dir + fid + '.txt',
                              ann_path=data_dir + fid + '.ann',
                              tokenizer=tokenizer)
        train_data.append(record)
        if verbose == 1:
            draw_progress_bar(idx + 1, split_idx)

    if verbose == 1:
        print('\n\nTest Data:')

    test_data = []
    for idx, fid in enumerate(test_ids):
        record = HealthRecord(fid, text_path=data_dir + fid + '.txt',
                              ann_path=data_dir + fid + '.ann',
                              tokenizer=tokenizer)
        test_data.append(record)
        if verbose == 1:
            draw_progress_bar(idx + 1, len(file_ids) - split_idx)

    return train_data, test_data


def read_ade_data(ade_data_dir: str = 'ade_data/',
                  train_ratio: int = 0.8,
                  verbose: int = 0) -> Tuple[List[Dict], List[Dict]]:
    """
    Reads train and test ADE data

    Parameters
    ----------

    ade_data_dir : str, optional
        Directory where the ADE data is located. The default is 'ade_data/'.

    train_ratio : int, optional
        Percentage split of train data. The default is 0.8.

    verbose : int, optional
        1 to print reading progress, 0 otherwise. The default is 0.

    Returns
    -------
    Tuple[List[Dict], List[Dict]]
        Train ADE data, Test ADE data.

    """

    # Get all the IDs of ADE data
    ade_file_ids = sorted(list(set(['.'.join(fname.split('.')[:-1]) \
                                    for fname in os.listdir(ade_data_dir) \
                                    if not fname.startswith('.')])))

    # Load ADE data
    ade_data = []
    for idx, fid in enumerate(ade_file_ids):
        with open(ade_data_dir + fid + '.json') as f:
            data = json.load(f)
            ade_data.extend(data)

    random.seed(0)
    random.shuffle(ade_data)

    ade_split_idx = int(train_ratio * len(ade_data))

    ade_train_data = process_ade_files(ade_data[:ade_split_idx])
    if verbose == 1:
        print("\n\nADE Train data:")
        print("Done.")

    ade_test_data = process_ade_files(ade_data[ade_split_idx:])
    if verbose == 1:
        print("\nADE Test data:")
        print("Done.")

    return ade_train_data, ade_test_data


def process_ade_files(ade_data: List[dict]) -> List[dict]:
    """
    Extracts tokens and creates Entity and Relation objects
    from raw json data.

    Parameters
    ----------
    ade_data : List[dict]
        Raw json data.

    Returns
    -------
    List[dict]
        Tokens, entities and relations.

    """
    ade_records = []

    for ade in ade_data:
        entities = {}
        relations = {}
        relation_backlog = []

        # Tokens
        tokens = ade['tokens']

        # Entities
        e_num = 1
        for ent in ade['entities']:
            ent_id = 'T' + "%s" % e_num
            ent_obj = Entity(entity_id=ent_id,
                             entity_type=ent['type'])

            r = [ent['start'], ent['end'] - 1]
            r = list(map(int, r))
            ent_obj.set_range(r)

            text = ''
            for token_ent in ade['tokens'][ent['start']:ent['end']]:
                text += token_ent + ' '
            ent_obj.set_text(text)

            entities[ent_id] = ent_obj
            e_num += 1

            # Relations
        r_num = 1
        for relation in ade['relations']:
            rel_id = 'R' + "%s" % r_num
            rel_details = 'ADE-Drug'
            entity1 = "T" + str(relation['head'] + 1)
            entity2 = "T" + str(relation['tail'] + 1)

            if entity1 in entities and entity2 in entities:
                rel = Relation(relation_id=rel_id,
                               relation_type=rel_details,
                               arg1=entities[entity1],
                               arg2=entities[entity2])

                relations[rel_id] = rel

            else:
                relation_backlog.append([rel_id, rel_details,
                                         entity1, entity2])
            r_num += 1

        ade_records.append({"tokens": tokens, "entities": entities, "relations": relations})
    return ade_records


def map_entities(entities: List[Entity],
                 actual_relations: List[Relation] = None) \
        -> Union[List[Relation], List[Tuple[Relation, int]]]:
    """
    Maps each drug entity to all other non-drug entities in the list.

    Parameters
    ----------
    entities : List[Entity]
        List of entities.

    actual_relations : List[Relation], optional
        List of actual relations (for training data).
        The default is None.

    Returns
    -------
    Union[List[Relations], List[Tuple[Relation, int]]]
        List of mapped relations. If actual relations are specified,
        also returns a flag to indicate if it is an actual relation.

    """
    drug_entities = []
    non_drug_entities = []

    # Splitting each entity to drug and non-drug entities
    for ent in entities:
        if ent.name.lower() == "drug":
            drug_entities.append(ent)
        else:
            non_drug_entities.append(ent)

    relations = []
    i = 1

    # Mapping each drug entity to each non-drug entity
    for ent1 in drug_entities:
        for ent2 in non_drug_entities:
            rel = Relation(relation_id="R%d" % i,
                           relation_type="Drug-" + ent2.name,
                           arg1=ent1, arg2=ent2)
            relations.append(rel)
            i += 1

    if actual_relations is None:
        return relations

    # Maps each relation type to list of actual relations
    actual_rel_dict = defaultdict(list)
    for rel in actual_relations:
        actual_rel_dict[rel.name].append(rel)

    relation_flags = []
    flag = 0

    # Computes actual relation flags
    for rel in relations:
        for act_rel in actual_rel_dict[rel.name]:
            if rel == act_rel:
                flag = 1
                break

        relation_flags.append(flag)
        flag = 0

    return list(zip(relations, relation_flags))


def draw_progress_bar(current, total, string='', bar_len=20):
    """
    Draws a progress bar, like [====>    ] 40%

    Parameters
    ------------
    current: int/float
             Current progress

    total: int/float
           The total from which the current progress is made

    string: str
            Additional details to write along with progress

    bar_len: int
            Length of progress bar
    """
    percent = current / total
    arrow = ">"
    if percent == 1:
        arrow = ""
    # Carriage return, returns to the beginning of line to overwrite
    sys.stdout.write("\r")
    sys.stdout.write("Progress: [{:<{}}] {}/{}".format("=" * int(bar_len * percent) + arrow,
                                                       bar_len, current, total) + string)
    sys.stdout.flush()


def is_whitespace(char):
    """
    Checks if the character is a whitespace

    Parameters
    --------------
    char: str
          A single character string to check
    """
    # ord() returns unicode and 0x202F is the unicode for whitespace
    if char == " " or char == "\t" or char == "\r" or char == "\n" or ord(char) == 0x202F:
        return True
    else:
        return False


def is_punct(char):
    """
    Checks if the character is a punctuation

    Parameters
    --------------
    char: str
          A single character string to check
    """
    if char == "." or char == "," or char == "!" or char == "?" or char == '\\':
        return True
    else:
        return False


def save_pickle(file, variable):
    """
    Saves variable as a pickle file

    Parameters
    -----------
    file: str
          File name/path in which the variable is to be stored

    variable: object
              The variable to be stored in a file
    """
    if file.split('.')[-1] != "pkl":
        file += ".pkl"

    with open(file, 'wb') as f:
        dump(variable, f)
        print("Variable successfully saved in " + file)


def open_pickle(file):
    """
    Returns the variable after reading it from a pickle file

    Parameters
    -----------
    file: str
          File name/path from which variable is to be loaded
    """
    if file.split('.')[-1] != "pkl":
        file += ".pkl"

    with open(file, 'rb') as f:
        return load(f)

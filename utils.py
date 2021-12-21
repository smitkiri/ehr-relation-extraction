from typing import List, Tuple, Callable, Dict, Union, Iterable
from annotations import Entity, Relation
from ehr import HealthRecord

import os
import sys
from pickle import dump, load
from IPython.core.display import display, HTML
import json
from collections import defaultdict
import pandas as pd
import networkx as nx
import math
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib


TPL_HTML = """<span style = "background-color: {color}; border-radius: 5px;">&nbsp;{content}&nbsp;</span>"""

TPL_HTML_HOVER = """<span style = "background-color: {color}; border-radius: 5px;" class="{grp}">&nbsp;{content}&nbsp;<span style = "background: {color}">{ent_type}</span></span>"""

COLORS = {"Drug": "#aa9cfc", "Strength": "#ff9561",
          "Form": "#7aecec", "Frequency": "#9cc9cc",
          "Route": "#ffeb80", "Dosage": "#bfe1d9",
          "Reason": "#e4e7d2", "ADE": "#ff8197",
          "Duration": "#97c4f5"}


def add_ent_group(entities: Union[Dict[str, Entity], List[Entity]],
                  relations: Union[Dict[str, Relation], List[Relation]]) -> List[Entity]:
    """
    Adds relation group to Entity objects.

    Parameters
    ----------
    entities : Union[Dict[str, Entity], List[Entity]]
        Entities

    relations : Union[Dict[str, Relation], List[Relation]])
        Relations

    Returns
    -------
    List[Entity]
        List of Entities with group information added.
    """

    # Convert entities to a dictionary if not
    if not isinstance(entities, dict):
        ent_dict = {}
        for ent in entities:
            ent_dict[ent.ann_id] = ent
        entities = ent_dict

    # Add group information
    for rel in relations:
        entities[rel.arg1.ann_id].relation_group += "group-" + rel.ann_id + " "
        entities[rel.arg2.ann_id].relation_group += "group-" + rel.ann_id + " "

    return list(entities.values())


# noinspection PyTypeChecker
def display_ehr(text: str,
                entities: Union[Dict[str, Entity], List[Entity]],
                relations: Union[Dict[str, Relation], List[Relation]] = None,
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

    relations : Union[Dict[str, Relation], List[Relation]]
        A list of relations. If provided, entities should be a dictionary.

    return_html : bool
        Indicator for returning HTML or printing the tagged EHR.
        The default is False.

    Returns
    -------
    Union[None, str]
        If return_html is true, returns html strings
        otherwise displays HTML.

    """
    if relations is not None:
        entities = add_ent_group(entities, relations)

    if isinstance(entities, dict):
        entities = list(entities.values())

    # Sort entity by starting range
    entities.sort(key=lambda x: x.range[0])

    # Final text to render
    render_text = ""
    start_idx = 0

    # Display legend
    if not return_html:
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
                grp=ent.relation_group,
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
        display(HTML(render_text))


def display_knowledge_graph(long_relation_df: pd.DataFrame, num_col: int = 2,
                            height: int = 8, width: int = 8,
                            return_html: bool = False) -> Union[None, str]:
    """
    Highlights EHR records with colors and displays
    them as HTML. Ideal for working with Jupyter Notebooks

    Parameters
    ----------
    long_relation_df: pd.DataFrame
        Relation dataframe in long format. Should have columns named:
        ['drug_id', 'drug', 'arg', 'edge']

    num_col: int
        Number of columns in the grid. Number of rows are automatically
        calculated based on this. The default is 2.

    height: int
        The height of a single graph in inches. The default is 6.

    width: int
        The width of a single graph in inches. The default is 6.

    return_html: bool
        Indicator for returning the HTML img tag or displaying the plot.
        The default is False.

    Returns
    -------
    Union[None, str]
        If return_html is true, returns html string
        otherwise displays the plot.

    """
    if return_html:
        matplotlib.use('Agg')

    drug_ids = sorted(list(pd.unique(long_relation_df['drug_id'])))
    num_row = math.ceil(len(drug_ids) / num_col)

    if num_row == 0:
        return None

    _ = plt.subplots(num_row, num_col, figsize=(num_col * width, height * num_row))

    i = 0
    for i, d in enumerate(drug_ids):
        sub_rel = long_relation_df[long_relation_df["drug_id"] == d]
        labels = sub_rel.set_index(['drug', 'arg'])['edge'].to_dict()

        plt.subplot(num_row, num_col, i + 1)

        # Knowledge graph for a single drug
        graph = nx.from_pandas_edgelist(sub_rel, "drug", "arg", edge_attr=True, create_using=nx.MultiDiGraph())

        # Drug will always be the first in the graph
        color_map = ['#aa9cfc'] + ['skyblue'] * (len(graph.nodes) - 1)

        pos = nx.spring_layout(graph)

        # Draw the graph
        nx.draw(graph, with_labels=True, font_size=12, pos=pos,
                node_color=color_map, node_size=2000)

        # Draw edge labels
        nx.draw_networkx_edge_labels(graph, edge_labels=labels,
                                     pos=pos, font_color='red')

    # Remove axis for empty plots, if any
    i += 1
    while i < num_row * num_col:
        plt.subplot(num_row, num_col, i + 1)
        plt.axis('off')
        i += 1

    if not return_html:
        plt.show()
        return

    # Create an encoding for the image
    tmp_file = BytesIO()

    plt.tight_layout()
    plt.savefig(tmp_file, format="png")

    encoded = base64.b64encode(tmp_file.getvalue()).decode('utf-8')
    img_tag = '<img id="knowledge-graph" src=\'data:image/png;base64,{}\'>'.format(encoded)

    return img_tag


def read_data(data_dir: str = 'data/',
              tokenizer: Callable[[str], List[str]] = None,
              is_bert_tokenizer: bool = True,
              verbose: int = 0) -> Tuple[List[HealthRecord], List[HealthRecord]]:
    """
    Reads train and test data

    Parameters
    ----------
    data_dir : str, optional
        Directory where the data is located.
        It should have directories named 'train' and 'test'
        The default is 'data/'.

    tokenizer : Callable[[str], List[str]], optional
        The tokenizer function to use.. The default is None.

    is_bert_tokenizer : bool
        If the tokenizer is a BERT-based WordPiece tokenizer

    verbose : int, optional
        1 to print reading progress, 0 otherwise. The default is 0.

    Returns
    -------
    Tuple[List[HealthRecord], List[HealthRecord]]
        Train data, Test data.

    """
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    # Get all IDs for train and test data
    train_ids = list(set(['.'.join(fname.split('.')[:-1]) \
                          for fname in os.listdir(train_path) \
                          if not fname.startswith('.')]))

    test_ids = list(set(['.'.join(fname.split('.')[:-1]) \
                         for fname in os.listdir(test_path) \
                         if not fname.startswith('.')]))

    if verbose == 1:
        print("Train data:")

    train_data = []
    for idx, fid in enumerate(train_ids):
        record = HealthRecord(fid, text_path=os.path.join(train_path, fid + '.txt'),
                              ann_path=os.path.join(train_path, fid + '.ann'),
                              tokenizer=tokenizer,
                              is_bert_tokenizer=is_bert_tokenizer)
        train_data.append(record)
        if verbose == 1:
            draw_progress_bar(idx + 1, len(train_ids))

    if verbose == 1:
        print('\n\nTest Data:')

    test_data = []
    for idx, fid in enumerate(test_ids):
        record = HealthRecord(fid, text_path=os.path.join(test_path, fid + '.txt'),
                              ann_path=os.path.join(test_path, fid + '.ann'),
                              tokenizer=tokenizer,
                              is_bert_tokenizer=is_bert_tokenizer)
        test_data.append(record)
        if verbose == 1:
            draw_progress_bar(idx + 1, len(test_ids))

    return train_data, test_data


def read_ade_data(ade_data_dir: str = 'ade_data/',
                  verbose: int = 0) -> List[Dict]:
    """
    Reads train and test ADE data

    Parameters
    ----------

    ade_data_dir : str, optional
        Directory where the ADE data is located. The default is 'ade_data/'.

    verbose : int, optional
        1 to print reading progress, 0 otherwise. The default is 0.

    Returns
    -------
    List[Dict]
        ADE data

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

    ade_data = process_ade_files(ade_data)
    if verbose == 1:
        print("\n\nADE data: Done")

    return ade_data


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
            if ent['type'] == 'Adverse-Effect':
                ent['type'] = 'ADE'

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


def map_entities(entities: Union[Dict[str, Entity], List[Entity]],
                 actual_relations: Union[Dict[str, Relation], List[Relation]] = None) \
        -> Union[List[Tuple[Relation, None]], List[Tuple[Relation, int]]]:
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

    if isinstance(entities, dict):
        entities = list(entities.values())

    if actual_relations and isinstance(actual_relations, dict):
        actual_relations = list(actual_relations.values())

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
                           relation_type=ent2.name + "-Drug",
                           arg1=ent1, arg2=ent2)
            relations.append(rel)
            i += 1

    if actual_relations is None:
        return list(zip(relations, [None] * len(relations)))

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


def get_long_relation_table(relations: Iterable[Relation]) -> pd.DataFrame:
    """
    Returns the relations in a long table format with the columns
    ['drug_id', 'drug', 'arg', 'edge'] where arg is entity related
    to drug and edge is the entity type.

    Parameters
    ----------
    relations : Iterable[Relation]
        A list of relations.

    Returns
    -------
    pd.DataFrame
        All the relations in a long tabular format.

    """
    rel_dict = {'drug_id': [], 'drug': [], 'arg': [], 'edge': []}

    for rel in relations:
        if rel.arg1.name == "Drug":
            rel_dict['drug_id'].append(rel.arg1.ann_id)
            rel_dict['drug'].append(rel.arg1.ann_text)
            rel_dict['arg'].append(rel.arg2.ann_text)

        else:
            rel_dict['drug_id'].append(rel.arg2.ann_id)
            rel_dict['drug'].append(rel.arg2.ann_text)
            rel_dict['arg'].append(rel.arg1.ann_text)

        rel_dict['edge'].append(rel.name.split('-')[0])

    rel_df = pd.DataFrame(rel_dict)
    return rel_df


def get_relation_table(relations: Union[pd.DataFrame, Iterable[Relation]],
                       is_long_df: bool = True) -> pd.DataFrame:
    """
    Returns the relations in a wide table format.

    Parameters
    ----------
    relations : Union[pd.DataFrame, Iterable[Relation]]
        Either a list of relations, or relations table in long format.

    is_long_df : bool
        Indicator for relations parameter. True indicates the input is
        a long dataframe. False indicates it is a list of relations.

    Returns
    -------
    str
        HTML blob of all the relations in a tabular format.

    """
    relations = relations.drop_duplicates()

    if not is_long_df:
        relations = get_long_relation_table(relations)

    relations = relations.rename(columns={"drug_id": "Drug ID", "drug": "Drug",
                                          "edge": "Entity Type", "arg": "Entity Text"})

    relation_df = (
        relations
        .groupby(["Drug ID", "Drug", "Entity Type"])["Entity Text"]
        .apply(lambda x: list(x))
        .reset_index(name="Entity Text")
        .set_index(["Drug ID", "Drug", "Entity Type"])
    )

    relation_df["Entity Text"] = relation_df["Entity Text"].apply(lambda x: "\n".join(x))

    empty_header = "    <tr style=\"text-align: right;\">\n      <th></th>\n      <th></th>\n      <th></th>\n      <th>Entity Text</th>\n    </tr>\n"
    empty_colname = "<th></th>"

    relation_html = (
        relation_df
        .to_html(classes=['table'], border=0)
        .replace("\\n", "<br>")
        .replace(empty_header, "")
        .replace(empty_colname, "<th>Entity Text</th>")
    )
    return relation_html


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

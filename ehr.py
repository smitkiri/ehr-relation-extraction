from annotations import Entity, Relation
from typing import List, Dict, Union, Tuple, Callable, Optional
import warnings
import numpy


class HealthRecord:
    """
    Objects that represent a single electronic health record
    """

    def __init__(self, record_id: str = "1", text_path: Optional[str] = None,
                 ann_path: Optional[str] = None,
                 text: Optional[str] = None,
                 tokenizer: Callable[[str], List[str]] = None,
                 is_bert_tokenizer: bool = True,
                 is_training: bool = True) -> None:
        """
        Initializes a health record object

        Parameters
        ----------
        record_id : int
            A unique ID for the record.

        text_path : str
            Path for the ehr record txt file.

        ann_path : str, optional
            Path for the annotation file. The default is None.

        text: str
            If text_path is not specified, the actual text for the
            record

        tokenizer: Callable[[str], List[str]], optional
            The tokenizer function to use. The default is None.

        is_bert_tokenizer: bool
            If the tokenizer is a BERT-based wordpiece tokenizer.
            The default is False.

        is_training : bool, optional
            Specifies if the record is a training example.
            The default is True.
        """
        if is_training and ann_path is None:
            raise AttributeError("Annotation path needs to be "
                                 "specified for training example.")

        if text_path is None and text is None:
            raise AttributeError("Either text or text path must be "
                                 "specified.")

        self.record_id = record_id
        self.is_training = is_training

        if text_path is not None:
            self.text = self._read_ehr(text_path)
        else:
            self.text = text

        self.char_to_token_map: List[int] = []
        self.token_to_char_map: List[int] = []
        self.tokenizer = None
        self.is_bert_tokenizer = is_bert_tokenizer
        self.elmo = None
        self.set_tokenizer(tokenizer)
        self.split_idx = None

        if ann_path is not None:
            annotations = self._extract_annotations(ann_path)
            self.entities, self.relations = annotations

        else:
            self.entities = None
            self.relations = None

    @staticmethod
    def _read_ehr(path: str) -> str:
        """
        Internal function to read EHR data.

        Parameters
        ----------
        path : str
            Path for EHR record.

        Returns
        -------
        str
            EHR record as a string.
        """
        f = open(path)
        raw_data = f.read()
        f.close()
        return raw_data

    @staticmethod
    def _extract_annotations(path: str) \
            -> Tuple[Dict[str, Entity], Dict[str, Relation]]:
        """
        Internal function that extracts entities and relations
        as a dictionary from an annotation file.

        Parameters
        ----------
        path : str
            Path for the ann file.

        Returns
        -------
        Tuple[Dict[str, Entity], Dict[str, Relation]]
            Entities and relations.
        """
        f = open(path)
        raw_data = f.read().split('\n')
        f.close()

        entities = {}
        relations = {}

        # Relations with entities that haven't been processed yet
        relation_backlog = []

        for line in raw_data:
            if line.startswith('#'):
                continue

            line = line.split('\t')

            # Remove empty strings from list
            line = list(filter(None, line))

            if not line or not line[0]:
                continue

            if line[0][0] == 'T':
                assert len(line) == 3

                idx = 0
                # Find the end of first word, which is the entity type
                for idx in range(len(line[1])):
                    if line[1][idx] == ' ':
                        break

                char_ranges = line[1][idx + 1:]

                # Get all character ranges, separated by ;
                char_ranges = [r.split() for r in char_ranges.split(';')]

                # Create an Entity object
                ent = Entity(entity_id=line[0],
                             entity_type=line[1][:idx])

                r = [char_ranges[0][0], char_ranges[-1][1]]
                r = list(map(int, r))
                ent.set_range(r)

                ent.set_text(line[2])
                entities[line[0]] = ent

            elif line[0][0] == 'R':
                assert len(line) == 2

                rel_details = line[1].split(' ')
                entity1 = rel_details[1].split(':')[-1]
                entity2 = rel_details[2].split(':')[-1]

                if entity1 in entities and entity2 in entities:
                    rel = Relation(relation_id=line[0],
                                   relation_type=rel_details[0],
                                   arg1=entities[entity1],
                                   arg2=entities[entity2])

                    relations[line[0]] = rel
                else:
                    # If the entities aren't processed yet, 
                    # add them to backlog to process later
                    relation_backlog.append([line[0], rel_details[0],
                                             entity1, entity2])

            else:
                # If the annotation is not a relation or entity, warn user
                msg = f"Invalid annotation encountered: {line}, File: {path}"
                warnings.warn(msg)

        for r in relation_backlog:
            rel = Relation(relation_id=r[0], relation_type=r[1],
                           arg1=entities[r[2]], arg2=entities[r[3]])

            relations[r[0]] = rel

        return entities, relations

    def _compute_tokens(self) -> None:
        """
        Computes the tokens and character <-> token index mappings
        for EHR text data.
        """
        self.tokens = list(map(lambda x: str(x), self.tokenizer(self.text)))

        char_to_token_map = []
        token_to_char_map = []

        j = 0
        k = 0

        for i in range(len(self.tokens)):
            # For BioBERT, a split within a word is denoted by ##
            if self.is_bert_tokenizer and self.tokens[i].startswith("##"):
                k += 2

            # Characters that are discarded from tokenization
            while self.text[j].lower() != self.tokens[i][k].lower():
                char_to_token_map.append(char_to_token_map[-1])
                j += 1

            # For SciSpacy, if there are multiple spaces, it removes
            # one and keeps the rest
            if self.text[j] == ' ' and self.text[j + 1] == ' ':
                char_to_token_map.append(char_to_token_map[-1])
                j += 1

            token_start_idx = j
            # Go over each letter in token and original text
            while k < len(self.tokens[i]):
                if self.text[j].lower() == self.tokens[i][k].lower():
                    char_to_token_map.append(i)
                    j += 1
                    k += 1
                else:
                    msg = f"Error computing token to char map. ID: {self.record_id}"
                    raise Exception(msg)

            token_end_idx = j
            token_to_char_map.append((token_start_idx, token_end_idx))
            k = 0

        # Characters at the end which are discarded by tokenizer
        while j < len(self.text):
            char_to_token_map.append(char_to_token_map[-1])
            j += 1

        assert len(char_to_token_map) == len(self.text)
        assert len(token_to_char_map) == len(self.tokens)

        self.char_to_token_map = char_to_token_map
        self.token_to_char_map = token_to_char_map

    def get_tokens(self) -> List[str]:
        """
        Returns the tokens.

        Returns
        -------
        List[str]
            List of tokens.
        """
        if self.tokenizer is None:
            raise AttributeError("Tokenizer not set.")

        return self.tokens

    def set_tokenizer(self, tokenizer: Callable[[str], List[str]]) \
            -> None:
        """
        Set the tokenizer for the object.

        Parameters
        ----------
        tokenizer : Callable[[str], List[str]]
            The tokenizer function to use.
        """
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self._compute_tokens()

    def get_token_idx(self, char_idx: int) -> int:
        """
        Returns the token index from character index.

        Parameters
        ----------
        char_idx : int
            Character index.

        Returns
        -------
        int
            Token index.
        """
        if self.tokenizer is None:
            raise AttributeError("Tokenizer not set.")

        token_idx = self.char_to_token_map[char_idx]

        return token_idx

    def get_char_idx(self, token_idx: int) -> int:
        """
        Returns the index for the first character of the specified
        token index.

        Parameters
        ----------
        token_idx : int
            Token index.

        Returns
        -------
        int
            Character index.
        """
        if self.tokenizer is None:
            raise AttributeError("Tokenizer not set.")

        char_idx = self.token_to_char_map[token_idx]

        return char_idx

    def get_labels(self) -> List[str]:
        """
        Get token labels in IOB format.

        Returns
        -------
        List[str]
            Labels.

        """
        if self.tokenizer is None:
            raise AttributeError("No tokens found. Set tokenizer first.")

        ent_label_map = {'Drug': 'DRUG', 'Strength': 'STR', 'Duration': 'DUR',
                         'Route': 'ROU', 'Form': 'FOR', 'ADE': 'ADE', 'Dosage': 'DOS',
                         'Reason': 'REA', 'Frequency': 'FRE'}

        labels = ['O'] * len(self.tokens)

        for ent in self.entities.values():
            start_idx = self.get_token_idx(ent.range[0])
            end_idx = self.get_token_idx(ent.range[1])

            for idx in range(start_idx, end_idx + 1):
                if idx == start_idx:
                    labels[idx] = 'B-' + ent_label_map[ent.name]
                else:
                    labels[idx] = 'I-' + ent_label_map[ent.name]

        return labels

    def get_split_points(self, max_len: int = 510,
                         new_line_ind: List[str] = None,
                         sent_end_ind: List[str] = None) -> List[int]:
        """
        Get the splitting points for tokens.

        > It includes as many paragraphs as it can within the
        max_len - 2 token limit. (2 less because BERT needs
                                  to add 2 special tokens)

        > If it can't find a single complete paragraph,
        it will split on the last verifiable new line that
        starts with a new sentence.

        > If it can't find that as well, it splits on token max_len - 2.

        Parameters
        ----------
        max_len : int, optional
            Maximum number tokens in one example. The default is 510
            for BERT.

        new_line_ind : List[str], optional
            New line indicators. Strings other than numbers.
            The default is ['[', '#', '-', '>', ' '].

        sent_end_ind : List[str], optional
            Sentence end indicators. The default is ['.', '?', '!'].

        Returns
        -------
        List[int]
            Splitting indices, includes the first and last index.
            Need to add 1 to the end indices if accessing
            with list splicing.

        """
        if new_line_ind is None:
            new_line_ind = ['[', '#', '-', '>', ' ']

        if sent_end_ind is None:
            sent_end_ind = ['.', '?', '!']

        split_idx = [0]
        last_par_end_idx = 0
        last_line_end_idx = 0

        for i in range(len(self.text)):
            curr_counter = self.get_token_idx(i) - split_idx[-1]

            if curr_counter >= max_len:
                # If not even a single paragraph has ended
                if last_par_end_idx == 0 and last_line_end_idx != 0:
                    split_idx.append(last_line_end_idx)

                elif last_par_end_idx != 0:
                    split_idx.append(last_par_end_idx)

                else:
                    split_idx.append(self.get_token_idx(i))

                last_par_end_idx = 0
                last_line_end_idx = 0

            if i < len(self.text) - 2 and self.text[i] == '\n':
                if self.text[i + 1] == '\n':
                    last_par_end_idx = self.get_token_idx(i - 1)

                if self.text[i + 1] == '.' or self.text[i + 1] == '*':
                    last_par_end_idx = self.get_token_idx(i + 1)

                if self.text[i + 1] in new_line_ind or \
                        self.text[i + 1].isdigit() or \
                        self.text[i - 1] in sent_end_ind:
                    last_line_end_idx = self.get_token_idx(i)

        split_idx.append(len(self.tokens))
        self.split_idx = split_idx

        return self.split_idx

    def get_annotations(self) -> Dict[str, Union[list, dict]]:
        """
        Get entities and relations in a dictionary.
        Entities are referenced with the key 'entities'
        and relations with 'relations'

        Returns
        -------
        Dict[Dict[str, Entity], Dict[str, Relation]]
            Entities and relations.
        """
        if self.entities is None or self.relations is None:
            raise AttributeError("Annotations not available")

        return {'entities': self.entities, 'relations': self.relations}

    def get_entities(self) -> Dict[str, Entity]:
        """
        Get the entities.

        Returns
        -------
        Dict[str, Entity]
            Entity ID: Entity object.
        """
        if self.entities is None:
            raise AttributeError("Entities not set")

        return self.entities

    def get_relations(self) -> Dict[str, Relation]:
        """
        Get the entity relations.

        Returns
        -------
        Dict[str, Relation]
            Relation ID: Relation Object.
        """
        if self.relations is None:
            raise AttributeError("Relations not set")

        return self.relations

    def _compute_elmo_embeddings(self) -> None:
        """
        Computes the Elmo embeddings for each token in EHR text data.
        """
        # noinspection PyUnresolvedReferences
        elmo_embeddings = self.elmo.embed_sentence(self.tokens)[-1]
        self.elmo_embeddings = elmo_embeddings

    def set_elmo_embedder(self, elmo: Callable[[str], numpy.ndarray]) -> None:
        """
        Set Elmo embedder for object.

        Parameters
        ----------
        elmo :
            The Elmo embedder to use.
        """
        self.elmo = elmo
        if elmo is not None:
            self._compute_elmo_embeddings()

    def get_elmo_embeddings(self) -> numpy.ndarray:
        """
        Get the elmo embeddings.

        Returns
        -------
        List[int]:
            Elmo embeddings for each word

        """
        if self.elmo_embeddings is None:
            raise AttributeError("Elmo embeddings not set")

        return self.elmo_embeddings

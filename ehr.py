from annotations import Entity, Relation
from typing import List, Dict, TypedDict, Tuple, Callable
import warnings
import re


class AnnotationInfo(TypedDict):
    '''
    Type hint for Annotations
    '''
    entities: Dict[str, Entity]
    relations : Dict[str, Relation]


class HealthRecord:
    '''
    Objects that represent a single electronic health record
    '''
    def __init__(self, record_id: str, text_path: str, 
                 ann_path: str = None,
                 tokenizer: Callable[[str], List[str]] = None,
                 is_training: bool = True) -> None:
        '''
        Initializes a health record object

        Parameters
        ----------
        record_id : int
            A unique ID for the record.
        
        text_path : str
            Path for the ehr record txt file.
        
        ann_path : str, optional
            Path for the annotation file. The default is None.
        
        tokenizer: Callable[[str], List[str]], optional
            The tokenizer function to use. The default is None.
        
        is_training : bool, optional
            Specifies if the record is a training example. 
            The default is True.
        '''
        if is_training and ann_path is None:
            raise AttributeError("Annotation path needs to be "
                                 "specified for training example.")
        
        self.record_id = record_id
        self.text = self._read_ehr(text_path)
        self.is_training = is_training
        self.set_tokenizer(tokenizer)
        
        self.char_to_word_map: List[int] = []
        self.word_to_token_map: List[int] = []
        self.token_to_word_map: List[int] = []
        
        if ann_path is not None:
            annotations = self._extract_annotations(ann_path)
            self.entities, self.relations = annotations
            self._compute_char_to_word_idx()
        
        else:
            self.entities = None
            self.relations = None
        
    
    def _read_ehr(self, path: str) -> str:
        '''
        Internal function to read EHR data.

        Parameters
        ----------
        path : str
            Path for EHR record.

        Returns
        -------
        str
            EHR record as a string.
        '''
        f = open(path)
        raw_data = f.read()
        f.close()
        return raw_data
    
    
    def _extract_annotations(self, path: str)\
        -> Tuple[Dict[str, Entity], Dict[str, Relation]]:
        '''
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
        '''
        f = open(path)
        raw_data = f.read().split('\n')
        f.close()
        
        entities = {}
        relations = {}
        
        # Relations with entities that haven't been processed yet
        relation_backlog = []
        
        for line in raw_data:
            line = line.split('\t')
            
            # Remove empty strings from list
            line = list(filter(None, line))
            
            if not line or not line[0]:
                continue
            
            if line[0][0] == 'T':
                assert len(line) == 3
                
                # Find the end of first word, which is the entity type
                for idx in range(len(line[1])):
                    if line[1][idx] == ' ':
                        break
                
                char_ranges = line[1][idx + 1:]
                
                # Get all character ranges, separated by ;
                char_ranges = [r.split() for r in char_ranges.split(';')]
                
                # Create an Entity object
                ent = Entity(entity_id = line[0], 
                             entity_type = line[1][:idx])
            
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
                    rel = Relation(relation_id = line[0], 
                                   relation_type = rel_details[0], 
                                   arg1 = entities[entity1], 
                                   arg2 = entities[entity2])
                    
                    relations[line[0]] = rel
                else:
                    # If the entities aren't processed yet, 
                    # add them to backlog to process later
                    relation_backlog.append([line[0], rel_details[0], 
                                             entity1, entity2])
            
            else:
                # If the annotation is not a relation or entity, warn user
                warnings.warn("Invalid annotation encountered: " + str(line))
        
        for r in relation_backlog:
            rel = Relation(relation_id = r[0], relation_type = r[1], 
                           arg1 = entities[r[2]], arg2 = entities[r[3]])
            
            relations[r[0]] = rel
            
        return entities, relations
    

    def _compute_char_to_word_idx(self) -> None:
        '''
        Internal function that computes character to word index map.

        It is a list which maps each character index to a word index
        Length of this list is equal to the number of characters
        in the text.
        '''
        char_to_word = []
        words = re.split('\n| |\t', self.text)

        for idx in range(len(words)):
            # The space next to a word will be considered as part of that word itself
            char_to_word = char_to_word + [idx] * (len(words[idx]) + 1)

        # There is no space after last word, so we need to remove the last element
        char_to_word = char_to_word[:-1]

        # Check for errors
        assert len(char_to_word) == len(self.text)

        self.char_to_word_map = char_to_word
    
    
    def _compute_tokens(self) -> None:
        '''
        Computes the tokens for EHR text data.
        '''
        token_to_org_map = []
        org_to_token_map = []
        all_doc_tokens = []
        
        words = re.split('\n| |\t', self.text)
    
        for idx, word in enumerate(words):
            org_to_token_map.append(len(all_doc_tokens))
            sub_tokens = self.tokenizer(word)
            # See if there are sub-tokens for the space-seperated word
            for token in sub_tokens:
                token_to_org_map.append(idx)
                all_doc_tokens.append(token.text)
        
        self.token_to_word_map = token_to_org_map
        self.word_to_token_map = org_to_token_map
        self.tokens = all_doc_tokens
        
    
    def get_tokens(self) -> List[str]:
        '''
        Returns the tokens.

        Returns
        -------
        List[str]
            List of tokens.
        '''
        if self.tokenizer is None:
            raise AttributeError("Tokenizer not set.")
            
        return self.tokens
        
        
    def set_tokenizer(self, tokenizer: Callable[[str], List[str]])\
            -> None:
        '''
        Set the tokenizer for the object.

        Parameters
        ----------
        tokenizer : Callable[[str], List[str]]
            The tokenizer function to use.
        '''
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self._compute_tokens()
        
    def get_token_idx(self, char_idx: int) -> int:
        '''
        Returns the token index from character index.

        Parameters
        ----------
        char_idx : int
            Character index.

        Returns
        -------
        int
            Token index.
        '''
        if self.tokenizer is None:
            raise AttributeError("Tokenizer not set.")
        
        word_idx = self.char_to_word_map[char_idx]
        token_idx = self.word_to_token_map[word_idx]
        
        #token_idx = self.char_to_token_map[char_idx]
        
        return token_idx
    
    
    def get_char_idx(self, token_idx: int) -> int:
        '''
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
        '''
        if self.tokenizer is None:
            raise AttributeError("Tokenizer not set.")
        
        word_idx = self.token_to_word_map[token_idx]
        char_idx = self.char_to_word_map.index(word_idx)
        
        #char_idx = self.char_to_token_map.index(token_idx)
        
        return char_idx
    
    def get_annotations(self) -> AnnotationInfo:
        '''
        Get entities and relations in a dictionary.
        Entities are referenced with the key 'entities'
        and relations with 'relations'

        Returns
        -------
        Dict[Dict[str, Entity], Dict[str, Relation]]
            Entities and relations.
        '''
        if self.entities is None or self.relations is None:
            raise AttributeError("Annotations not available")
            
        return {'entities': self.entities, 'relations': self.relations}
    
    
    def get_entities(self) -> Dict[str, Entity]:
        '''
        Get the entities.

        Returns
        -------
        Dict[str, Entity]
            Entity ID: Entity object.
        '''
        if self.entities is None:
            raise AttributeError("Entities not set")
        
        return self.entities
    
    
    def get_relations(self) -> Dict[str, Relation]:
        '''
        Get the entity relations.

        Returns
        -------
        Dict[str, Relation]
            Relation ID: Relation Object.
        '''
        if self.relations is None:
            raise AttributeError("Relations not set")
        
        return self.relations
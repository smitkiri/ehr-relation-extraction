from __future__ import annotations

from typing import List
from ehr import HealthRecord
from annotations import Entity
from collections import defaultdict
import re


class DictNER:
    '''
    A dictionary based NER model.
    '''
    def __init__(self):
        self.ner_re: dict = {}

    def _get_clean_re(self, entity_list: List[str]) -> str:
        '''
        Generates a regular expression from a list of entities

        Parameters
        ----------
        entity_list : List[str]
            List of entity text.

        Returns
        -------
        entity_re : str
            Regular expression.

        '''
        regex_chars = ['(', ')', '[', ']', '{', '}', '+', '*', '?', '$', '^', '&']
        
        for i in range(len(entity_list)):
            # We need to add a \ so it does not take entity text as regex
            # character
            for char in regex_chars:
                entity_list[i] = entity_list[i].replace(char, 
                                                        '\\' + char)
        
        # A space/new line/tab before and after the text to indicate
        # a seperate word
        entity_re = '[\n| |\t]|[\n| |\t]'.join(entity_list)
        entity_re = '[\n| |\t]' + entity_re + '[\n| |\t]'
        
        return entity_re
        
    def fit(self, train_data: List[HealthRecord]) -> DictNER:
        '''
        Generates a dictionary for the model

        Parameters
        ----------
        train_data : List[HealthRecord]
            Records to generate the dictionary from.

        Returns
        -------
        DictNER
            Self object.

        '''
        ner_dict = defaultdict(list)
        
        for data in train_data:
            for ent in data.entities.values():
                # We have a specific RE for Dosage
                if ent.name != 'Strength':
                    # Ignore text with length 1
                    if ent.ann_text.lower() not in ner_dict[ent.name]\
                        and len(ent.ann_text) > 1: 
                        ner_dict[ent.name].append(ent.ann_text.lower())
        
        for name, entity_list in ner_dict.items():
            ner_dict[name] = self._get_clean_re(entity_list)
        
        # Dosage is just a number followed by mg or mcg
        ner_dict['Strength'] = '\d+[ ]*(?:mg|mcg)'
        self.ner_re = dict(ner_dict)
        return self
    
    def predict(self, test_data: List[HealthRecord])\
            -> List[List[Entity]]:
        '''
        Returns character ranges for all predicted entities

        Parameters
        ----------
        test_data : List[HealthRecord]
            Text to predict the entities.

        Returns
        -------
        List[List[Entity]]
            Predictions for each example. Each prediction list 
            contains several Entity objects.

        '''
        predictions = []
        for data in test_data:
            entities = []
            j = 1
            for ent_name, ent_re in self.ner_re.items():
                # Get the start and end character ranges of entities
                # Remove the extra space at the start and end of entity
                ranges = [(m.start(0) + 1, m.end(0) - 1, ent_name) \
                                      for m in re.finditer(ent_re, data.text, re.IGNORECASE)]
                 
                # Convert to Entity Objects
                for r in ranges:
                    ent = Entity(entity_id = "T" + str(j))
                    ent.set_range([r[0], r[1]])
                    ent.set_entity_type(r[2])
                    entities.append(ent)
                    j += 1
            
            predictions.append(entities)
                        
        return predictions

from __future__ import annotations

import torch
torch.manual_seed(0)

from torch import nn
import torch.nn.functional as F

from typing import List, Tuple
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
                if ent.name != 'Dosage':
                    # Ignore text with length 1
                    if ent.ann_text.lower() not in ner_dict[ent.name]\
                        and len(ent.ann_text) > 1: 
                        ner_dict[ent.name].append(ent.ann_text.lower())
        
        for name, entity_list in ner_dict.items():
            ner_dict[name] = self._get_clean_re(entity_list)
        
        # Dosage is just a number followed by mg or mcg
        ner_dict['Dosage'] = '\d+[ ]*(?:mg|mcg)'
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
                    ent.add_range([r[0], r[1]])
                    ent.set_entity_type(r[2])
                    entities.append(ent)
                    j += 1
            
            predictions.append(entities)
                        
        return predictions
    

class LinearBlock(nn.Module):
    '''
    Bi-LSTM + CRF based model
    '''
    def __init__(self, ni, nf, drop):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)

    def forward(self, x):
        return self.lin(self.drop(self.bn(x)))


class LinearClassifier(nn.Module):

    def __init__(self, config, layers, drops):
        self.config = config
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def forward(self, input):
        output = input
        sl,bs,_ = output.size()
        x = output.view(-1, 2*self.config.hidden_size_lstm)

        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x.view(sl, bs, self.config.ntags)


class BiLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dropout = nn.Dropout(p=self.config.dropout)

        self.word_lstm = nn.LSTM(
            self.config.dim_elmo,
            self.config.hidden_size_lstm,
            bidirectional=True
        )

        self.linear = LinearClassifier(
            self.config,
            layers=[self.config.hidden_size_lstm*2, self.config.ntags],
            drops=[0.5]
        )

    def forward(self, input):
        word_emb = self.dropout(input.transpose(0, 1))
        output, (h, c) = self.word_lstm(word_emb) #shape = S*B*hidden_size_lstm
        output = self.dropout(output)

        output = self.linear(output)
        return output #shape = S*B*ntags

# CRF: https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
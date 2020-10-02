from typing import List

class Annotation:
    '''
    A generic class for annotations
    '''
    def __init__(self, ann_id: str, ann_name: str = None) -> None:
        self.ann_id = ann_id
        self.name = ann_name
        

class Entity(Annotation):
    '''
    Objects that represent named entities
    '''
    def __init__(self, entity_id: str, 
                 entity_type: str = None) -> None:
        
        super().__init__(entity_id, entity_type)
        self.ranges: List[List[int]] = []
        self.ann_text: str = ""
        
    def add_range(self, new_range: List[int]) -> None:
        '''
        Add annotation range
        '''
        self.ranges.append(new_range)
        
    def set_text(self, text: str) -> None:
        '''
        Sets the annotation text
        '''
        self.ann_text = text
        
    def set_entity_type(self, entity_type: str) -> None:
        '''
        Sets the entity type
        '''
        self.name = entity_type
    
    def __repr__(self) -> str:
        '''
        String representation of the object
        '''
        string = "\n"
        string += "ID: " + self.ann_id + "\n"
        string += "Entity name: " + self.name + "\n"
        
        string += "Character ranges: "
        string += "; ".join([str(r[0]) + " " + str(r[1]) 
                             for r in self.ranges]) + "\n"     
        string += "Entity text: " + self.ann_text    
        return string
    
    def __str__(self) -> str:
        return self.__repr__()


class Relation(Annotation):
    '''
    Objects that represent relations between named entities
    '''
    def __init__(self, relation_id: str, relation_type: str, 
                 arg1: Entity = None, arg2: Entity = None) -> None:
        
        super().__init__(relation_id, relation_type)
        self.arg1 = arg1
        self.arg2 = arg2
        
    def set_entity_relation(self, arg1: str, arg2: str) -> None:
        '''
        Sets the entities that are related
        '''
        self.arg1 = arg1
        self.arg2 = arg2
        
    def get_entities(self) -> List[str]:
        '''
        Returns related entities
        '''
        return [self.arg1, self.arg2]
    
    def set_relation_type(self, relation_type: str) -> None:
        '''
        Sets the relation type
        '''
        self.name = relation_type
        
    def __repr__(self) -> str:
        '''
        String representation of the object
        '''
        string = "\n"
        string += "ID: " + self.ann_id + "\n"
        string += "Relation type: " + self.name + "\n"
        string += "\nEntity 1: \n"
        string += self.arg1.__repr__() + "\n"
        string += "\nEntity 2: \n"
        string += self.arg2.__repr__()
        
        return string
    
    def __str__(self) -> str:
        return self.__repr__()
from typing import List, Tuple, Optional, Dict, Union
import json


class Annotation:
    """
    A generic class for annotations
    """

    def __init__(self, ann_id: str, ann_name: str = None) -> None:
        self.ann_id = ann_id
        self.name = ann_name


class Entity(Annotation):
    """
    Objects that represent named entities
    """

    def __init__(self, entity_id: str,
                 entity_type: str = None,
                 char_range: List[int] = None) -> None:
        """
        Initializes Entity object.

        Parameters
        ----------
        entity_id : str
            Unique entity ID.
        entity_type : str, optional
            The type of entity. The default is None.

        """
        super().__init__(entity_id, entity_type)
        if char_range is None:
            self.range = [None, None]
        else:
            self.range: List[int] = char_range
        self.ann_text: str = ""
        self.relation_group: str = ""

    def set_range(self, new_range: List[int]) -> None:
        """
        Add annotation range
        """
        self.range = new_range

    def set_text(self, text: str) -> None:
        """
        Sets the annotation text
        """
        self.ann_text = text

    def set_entity_type(self, entity_type: str) -> None:
        """
        Sets the entity type
        """
        self.name = entity_type

    def __repr__(self) -> str:
        """
        String representation of the object
        """
        string = "\n"
        string += "ID: " + self.ann_id + "\n"
        string += "Entity name: " + str(self.name) + "\n"

        string += "Character range: "
        string += str(self.range[0]) + " " + str(self.range[1]) + "\n"

        if self.ann_text:
            string += "Entity text: '" + str(self.ann_text) + "'"

        return string

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, key: int) -> int:
        """
        Makes the class subsriptable on range
        """
        return self.range[key]

    def __iter__(self) -> Tuple[int, int]:
        """
        Makes class iterable on range
        """
        yield self.range[0]
        yield self.range[1]

    def __eq__(self, other) -> bool:
        """
        Overrides equality method
        """
        if self.name == other.name and self.range == other.range:
            return True
        else:
            return False

    def to_json(self, output_string: bool = True) -> Union[str, dict]:
        """
        Converts the object to a json format

        Parameters
        ----------
        output_string : bool
            Flag to indicate if the output should be a string.
            If False, the output will be a dictionary.

        Returns
        -------
        Union[str, dict]
            The representation of the object in json format
        """
        class_dict = {
            "ent_id": self.ann_id,
            "ent_name": self.name,
            "char_range": self.range,
            "text": self.ann_text,
            "rel_group": self.relation_group
        }

        if output_string:
            return json.dumps(class_dict)
        else:
            return class_dict

    @classmethod
    def from_json(cls, data_dict: Union[str, dict]):
        """
        Creates a class object from a dictionary object

        Parameters
        ----------
        data_dict : Union[str, dict]
            A dictionary with the following mandatory fields: ent_id, ent_name, char_range
            Optional fields: text, rel_group

        Returns
        -------
        Entity object
        """
        if isinstance(data_dict, str):
            data_dict = json.loads(data_dict)

        try:
            if isinstance(data_dict["char_range"], str):
                data_dict["char_range"] = eval(data_dict["char_range"])

            obj = cls(
                entity_id=data_dict["ent_id"],
                entity_type=data_dict["ent_name"],
                char_range=data_dict["char_range"]
            )

        except KeyError as key:
            msg = f"Could not find the key {key}. The dictionary should contain the following keys: "\
                  "ent_id, ent_name, char_range"
            raise KeyError(msg)

        if "text" in data_dict:
            obj.ann_text = data_dict["text"]

        if "rel_group" in data_dict:
            obj.relation_group = data_dict["rel_group"]

        return obj


class Relation(Annotation):
    """
    Objects that represent relations between named entities
    """

    def __init__(self, relation_id: str, relation_type: str,
                 arg1: Entity = None, arg2: Entity = None) -> None:

        super().__init__(relation_id, relation_type)
        self.arg1 = arg1
        self.arg2 = arg2

    def set_entity_relation(self, arg1: str, arg2: str) -> None:
        """
        Sets the entities that are related
        """
        self.arg1 = arg1
        self.arg2 = arg2

    def get_entities(self) -> List[Optional[Entity]]:
        """
        Returns related entities
        """
        return [self.arg1, self.arg2]

    def set_relation_type(self, relation_type: str) -> None:
        """
        Sets the relation type
        """
        self.name = relation_type

    def __repr__(self) -> str:
        """
        String representation of the object
        """
        string = "\n"
        string += "ID: " + str(self.ann_id) + "\n"
        string += "Relation type: " + str(self.name) + "\n"
        string += "\nEntity 1: \n"
        string += self.arg1.__repr__() + "\n"
        string += "\nEntity 2: \n"
        string += self.arg2.__repr__()

        return string

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other) -> bool:
        """
        Overrides the default equality method.
        """
        if self.arg1 == other.arg1 and self.arg2 == other.arg2:
            return True

        elif self.arg2 == other.arg1 and self.arg1 == other.arg2:
            return True

        else:
            return False

    def to_json(self, output_string: bool = True) -> Union[str, dict]:
        """
        Converts the object to a json format

        Parameters
        ----------
        output_string : bool
            Flag to indicate if the output should be a string.
            If False, the output will be a dictionary.

        Returns
        -------
        Union[str, dict]
            The representation of the object in json format
        """
        class_dict = {
            "rel_id": self.ann_id,
            "rel_name": self.name,
            "arg1": self.arg1.to_json(output_string=False),
            "arg2": self.arg2.to_json(output_string=False)
        }

        if output_string:
            return json.dumps(class_dict)
        else:
            return class_dict

    @classmethod
    def from_json(cls, data_dict: Union[str, dict]):
        """
        Create a Relation object from a dictionary.

        Parameters
        ----------
        data_dict : Union[str, dict]
            A dictionary with the following keys: rel_id, rel_name, arg1, arg2
            The arg1 and arg2 keys can contain either Entity objects or a dictionary
            which can be converted to an Entity object

        Returns
        -------
        A Relation object
        """
        if isinstance(data_dict, str):
            data_dict = json.loads(data_dict)

        try:
            if isinstance(data_dict["arg1"], dict):
                data_dict["arg1"] = Entity.from_json(data_dict["arg1"])

            if isinstance(data_dict["arg2"], dict):
                data_dict["arg2"] = Entity.from_json(data_dict["arg2"])

            obj = cls(
                relation_id=data_dict["rel_id"],
                relation_type=data_dict["rel_name"],
                arg1=data_dict["arg1"],
                arg2=data_dict["arg2"]
            )

        except KeyError as key:
            msg = f"Could not find the key {key}. The dictionary should contain the following keys: "\
                  "rel_id, rel_name, arg1, arg2"
            raise KeyError(msg)

        return obj

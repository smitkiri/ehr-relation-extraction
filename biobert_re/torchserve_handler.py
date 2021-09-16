from abc import ABC
import logging
from json import JSONDecodeError

from utils_re import RETestDataset
from ehr import HealthRecord
from annotations import Relation

from typing import Union, Sequence, List, Tuple

from transformers import (AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer,
                          TrainingArguments, Trainer)
import numpy as np


from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
BIOBERT_RE_SEQ_LEN = 128


class REHandler(BaseHandler, ABC):
    """
    Torchserve handler for the RE task
    """
    def __init__(self):
        super(REHandler, self).__init__()
        self.initialized = False

        self.tokenizer: Union[PreTrainedTokenizer, None] = None
        self.label_list: Union[Sequence[str], None] = None
        self.training_args: Union[TrainingArguments, None] = None
        self.trainer: Union[Trainer, None] = None

    def initialize(self, context):
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        logger.info("Transformer model from path %s loaded successfully", model_dir)

        self.label_list = ["0", "1"]
        self.training_args = TrainingArguments(output_dir="/tmp", do_predict=True)
        self.trainer = Trainer(model=self.model, args=self.training_args)

        self.initialized = True

    def preprocess(self, data) -> Tuple[HealthRecord, RETestDataset]:
        input_data = data[0].get("data")
        if input_data is None:
            input_data = data[0].get("body")

        if isinstance(input_data, (bytes, bytearray)):
            input_data = input_data.decode("utf-8")

        # Create a HealthRecord object from the JSON input
        test_ehr = HealthRecord.from_json(input_data)
        test_ehr.tokenizer = self.tokenizer

        # Create a dataset object for input to the model
        test_dataset = RETestDataset(
            test_ehr=test_ehr,
            tokenizer=self.tokenizer,
            max_seq_len=BIOBERT_RE_SEQ_LEN,
            label_list=self.label_list
        )

        return test_ehr, test_dataset

    def inference(self, data: RETestDataset, *args, **kwargs) -> List[Relation]:
        if len(data) == 0:
            return []

        # Get the binary predictions to indicate if the relations exist
        re_predictions = self.trainer.predict(test_dataset=data).predictions
        re_predictions = np.argmax(re_predictions, axis=1)

        # Add the relations where the model predicted 1 to the relation list of the HealthRecord object
        idx = 1
        rel_preds = []
        for relation, pred in zip(data.relation_list, re_predictions):
            if pred == 1:
                relation.ann_id = "R%d" % idx
                idx += 1
                rel_preds.append(relation)

        return rel_preds

    def postprocess(self, data: dict) -> HealthRecord:
        # Set the relations attribute in the HealthRecord object to the predicted relations
        data["test_ehr"].relations = data["predictions"]
        return data["test_ehr"]

    def handle(self, data, context):
        test_ehr, test_dataset = self.preprocess(data)
        rel_predictions = self.inference(test_dataset)
        output_ehr = self.postprocess({"predictions": rel_predictions, "test_ehr": test_ehr})

        # Send the output in a JSON format
        return [output_ehr.to_json()]

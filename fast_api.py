"""API to generate tags for each token in a given EHR"""

# To run, execute: uvicorn fast_api:app --reload

from fastapi import FastAPI
from pydantic import BaseModel

from predict import get_ner_predictions
from utils import display_ehr


class NERTask(BaseModel):
    ehr_text: str
    model_choice: str


app = FastAPI()


@app.post("/ner/")
def create_ehr(ner_input: NERTask):
    """Request EHR text data and the model choice for NER Task"""

    predictions = get_ner_predictions(
        ehr_record=ner_input.ehr_text,
        model_name=ner_input.model_choice
    )

    return display_ehr(
        text=ner_input.ehr_text,
        entities=predictions,
        return_html=True
    )

"""API to generate tags for each token in a given EHR"""

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from predict import get_ner_predictions, get_re_predictions
from utils import display_ehr, get_long_relation_table, display_knowledge_graph, get_relation_table

import os

TORCHSERVE_URL = os.getenv("TORCHSERVE_URL", "http://127.0.0.1:8080")
TORCHSERVE_MODEL_MAP = {"biobert": "biobert_ner_v1.0"}


class NERTask(BaseModel):
    ehr_text: str
    model_choice: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
def get_ehr_predictions(ner_input: NERTask):
    """Request EHR text data and the model choice for NER Task"""

    ner_predictions = get_ner_predictions(
        ehr_record=ner_input.ehr_text,
        model_name=ner_input.model_choice)

    re_predictions = get_re_predictions(ner_predictions)
    relation_table = get_long_relation_table(re_predictions.relations)

    html_ner = display_ehr(
        text=ner_input.ehr_text,
        entities=ner_predictions.get_entities(),
        relations=re_predictions.relations,
        return_html=True)

    graph_img = display_knowledge_graph(relation_table, return_html=True)
    
    if len(relation_table) > 0:
        relation_table_html = get_relation_table(relation_table)
    else:
        relation_table_html = "<p>No relations found</p>"

    if graph_img is None:
        graph_img = "<p>No Relation found!</p>"

    return {'tagged_text': html_ner, 're_table': relation_table_html, 'graph': graph_img}


@app.get("/sample/")
def get_sample_ehr():
    """Returns a sample EHR record"""
    with open("sample_ehr/sample.txt") as f:
        sample_ehr = f.read()

    return {"data": sample_ehr}

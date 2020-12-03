"""API to generate tags for each token in a given EHR"""

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from predict import get_ner_predictions, get_re_predictions
from utils import display_ehr, get_long_relation_table, display_knowledge_graph, get_relation_table


class NERTask(BaseModel):
    ehr_text: str
    model_choice: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://smitkiri.me", "https://smitkiri.github.io", "*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/")
def get_ehr_predictions(ner_input: NERTask):
    """Request EHR text data and the model choice for NER Task"""

    ehr_predictions = get_ner_predictions(
        ehr_record=ner_input.ehr_text,
        model_name=ner_input.model_choice)

    html_ner = display_ehr(
        text=ner_input.ehr_text,
        entities=ehr_predictions.get_entities(),
        return_html=True)

    predicted_relations = get_re_predictions(ehr_predictions)
    relation_table = get_long_relation_table(predicted_relations)

    graph_img = display_knowledge_graph(relation_table, return_html=True)
    
    if len(relation_table) > 0:
        relation_table_html = get_relation_table(relation_table)
    else:
        relation_table_html = "<p>No relations found</p>"

    if graph_img is None:
        graph_img = "<p>No Relation found!</p>"

    return {'tagged_text': html_ner, 're_table': relation_table_html, 'graph': graph_img}

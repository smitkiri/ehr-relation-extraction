# NER and Relation Extraction from EHR
This repository includes code for NER and RE methods on EHR records. These methods were performed on n2c2 challenge dataset which was augmented to include a sample of ADE corpus dataset. This project is still in progress, the readme includes all the completed sections of the project in detail.

- [How to Run](#how-to-run)
- [Introduction](#Introduction)
- [Named Entity Recognition](#Named-Entity-Recognition-(NER))
    - [Data Preprocessing](#Data-Preprocessing)
    - [Rule-based Model](#Rule-based-model)
    - [BiLSTM + CRF](#BiLSTM-+-CRF)
    - [BioBERT](#BioBERT)
    - [NER Results](#ner-results)
- [References](#References)

# How to Run
To generate the preprocessed data required for model input, run from terminal:
```
python generate_data.py \
	--input_dir data/ \
	--ade_dir ade_corpus/ \
	--target_dir dataset/ \
	--max_seq_len 512 \
	--dev_split 0.1 \
	--test_split 0.2 \
	--tokenizer biobert-base \
	--ext txt \
	--sep " " \
```
The ade_dir is an optional parameter, and others have default values set. 
<br>
Instructions for running individual models can be found in their respective directories.

# Introduction
An Electronic Health Record (EHR) [[1]](https://www.cms.gov/Medicare/E-Health/EHealthRecords) is an electronic version of a patient's medical history that includes extremely important information including, but not limited to, problems, medication, progress notes, immunizations and laboratory reports. EHRs are huge free-text data files that are documented by healthcare professionals, like clinical notes, discharge summaries or lab reports. Finding information from this data is time consuming, since the data is unstructured and there may be multiple such records for a single patient. Natural Language Processing (NLP) techniques could be used to make this data structured, and quickly find information whenever needed, thereby saving healthcare professionals' time from these mundane tasks.

In this project, we aim to build a tool that would automatically structure this data into a format that would enable doctors and patients to quickly find information that they need. Specifically, we aim to build a Named Entity Recognition (NER) model that would recognize entities such as drug, strength, duration, frequency, adverse drug event (ADE) [[2]](https://www.cdc.gov/medicationsafety/adult_adversedrugevents.html), reason for taking the drug, route and form. Further, the model would also recognize the relationship between drug and every other named entity as well. This would allow healthcare professionals to not only look at individual entities, but also all the relationships between them. This would also allow the doctors to easily find out the relationships between a drug and ADEs so that such drugs can be monitored carefully. \par

The final goal of this project is to build an API where healthcare professionals and patients could send EHR data and the API would return character ranges for each annotation so they can be highlighted in the original data, a structured json-format data that includes separately labelled data for medication history and discharge medications. The highlighted annotations could be useful when a healthcare professional wants to see important information along with other details in the EHR. The structured information can be used to store the data for quick reference in the future. Because the EHR contains medication history as well as discharge medications, labelling them as such could help in merging new information, as the medication history would remain the same.

# Named Entity Recognition (NER)
To perform NER on the data, three different models were built. A rule-based model was built as  a baseline and two machine learning models were built.

## Data Preprocessing
The EHR records are usually lengthy, and it is not desirable to have such big input sizes for machine learning models, especially for models like BERT that have an input size restriction of 512 tokens. So, a function was implemented that would split the EHR records based on a maximum sequence length parameter. The function tries to include maximum number of tokens, maintaining as much context as possible for every token. The splitting points are decided based on the following criteria:

1) Includes as many paragraphs as possible within the maximum token limit, and splits at the end of the last paragraph found.
2) If the function cannot find a single complete paragraph, it splits on the last line (within the token limit) which marks the end of a sentence.
3) Otherwise, the function includes as many tokens as specified by the token limit, and then splits on the next index.

The data is tokenized using a modified ScispaCy tokenizer for BiLSTM + CRF model which just removes the tokens with whitespace characters after ScispaCy tokenizes them. For BioBERT model, the BioBERT base tokenizer was used to tokenize the data. Each sequence of labels or tokens in the data was represented using the IOB2 (Inside, Outside, Beginning) tagging scheme for BioBERT and BiLSTM models.

## Rule-based Model
To establish a baseline, a traditional dictionary and regular-expression based NER model was used. A regular expression was written to find the dosage entity, which would find any number followed by "mg" or "mcg". For all other entities, the data was split into 80% train data and 20% test data. The train data was used to create a dictionary of each entities, so if the same entities appear in the test data, it would classify it as the corresponding entity.

## BiLSTM + CRF
Just a BiLSTM network is enough to classify each token into various entities along with it's class (i.e. B: beginning or I: inside) or if it is not a part of any of the entities we are looking for (O: outside) but we witnessed some common errors of misclassification. Because the outputs of BiLSTM of each word are the label scores, we can select the label which has the highest score for each word. By this scheme, we may end up with invalid outputs, for eg: I-Drug followed by I-ADE or B-Drug followed by I-ADE. Hence we use the CRF (Conditional Random Field) algorithm to calculate the loss of our BiLSTM network as it could add some constraints to the final predicted labels to ensure they are valid. These constraints can be learned by CRF automatically from the training dataset during the training process. CRFs considers the context as well rather than predicting label for a single token without considering neighboring samples [[3]](https://medium.com/ml2vec/overview-of-conditional-random-fields-68a2a20fa541). 

The model was built using the architecture described in Guillaume Genthial's Blog [[4]](https://guillaumegenthial.github.io/) and it's PyTorch implementation [[5]](https://github.com/yongyuwen/PyTorch-Elmo-BiLSTMCRF). To train the model, the EHR dataset was tokenized using a modified version of the ScispaCy [[6]](https://allenai.github.io/scispacy/) tokenizer. The original tokenizer keeps all whitespace characters in separate tokens, but it was modified so that all of the white space tokens are removed. Every other tokens would remain the same. The input sequence length was set to 512 and the EHR records were split by using the steps discussed in the preprocessing section. The model was trained for 15 epochs using GPU resources from Google Cloud compute engine.

## BioBERT
The output of each token from the BERT model is passed through a fully connected neural network with a softmax layer at the end that classify that token to an entity. The entities here would be in IOB format, for example B-DRUG and I-DRUG would be treated as separate entities. This entire model is called BERT for token classification, and it's architecture is available in python's transformers library [[7]](https://huggingface.co/transformers/).

BioBERT is a pre-trained BERT model, that is trained on medical corpra of more than 18 billion words. Since it has a medical vocabulary and is trained on biomedical data, we chose this model to fine tune on our dataset. Code for fine tuning from the official BioBERT for PyTorch GitHub repository [[8]](https://github.com/dmis-lab/biobert-pytorch/tree/master/named-entity-recognition) was used with modifications in input format. The input sequence length was set to 128, and the model was fine tuned for 5 epochs using GPU resources from Google Cloud compute engine.   

## NER Results

The Rule-Based model did not perform very well, but that was expected as it does not take context into account and has a very high false positive rate. For BiLSTM + CRF and BioBERT, a sample of an external dataset, the ADE-corpus dataset\cite{ade-corpus} was integrated to our data which improved the performance to a great extent. The F1 score for ADE entity improved from 0.3403 to 0.8673 for the BioBERT model after adding a sample of the ADE Corpus.

![F1 breakdown per entity](https://github.com/smitkiri/ehr-relation-extraction/blob/master/plots/ner_entity_scores.jpg?raw=true)

Also, the BioBERT and BiLSTM + CRF models produce F1 scores similar to that of the model that won the n2c2 challenge using the same data-set. The winning model was submitted by Alibaba Inc. and used an architecture of BiLSTM + CNN for character-level + CRF for dependencies.

![Challenge comparision](https://github.com/smitkiri/ehr-relation-extraction/blob/master/plots/ner_micro_f1_challenge.jpg?raw=true)

|Model|Micro F1|
|:---:|:---:|
|Rule Based|0.1432|
|BiLSTM + CRF|0.8577|
|BioBERT|0.8944|
|Alibaba Inc. (Challenge winner)|0.8956|

# References

[1] Electronic Health Records: [https://www.cms.gov/Medicare/E-Health/EHealthRecords](https://www.cms.gov/Medicare/E-Health/EHealthRecords)

[2] Adverse Drug Events: [https://www.cdc.gov/medicationsafety/adult_adversedrugevents.html](https://www.cdc.gov/medicationsafety/adult_adversedrugevents.html)

[3] Conditional Random Fields: [https://medium.com/ml2vec/overview-of-conditional-random-fields-68a2a20fa541](https://medium.com/ml2vec/overview-of-conditional-random-fields-68a2a20fa541)

[4] Guillaume Genthial's Blog: [https://guillaumegenthial.github.io/](https://guillaumegenthial.github.io/)

[5] PyTorch-ELMo-BiLSTM-CRF implementation: [https://github.com/yongyuwen/PyTorch-Elmo-BiLSTMCRF](https://github.com/yongyuwen/PyTorch-Elmo-BiLSTMCRF)

[6] ScispaCy: [https://allenai.github.io/scispacy/](https://allenai.github.io/scispacy/)

[7] HuggingFace Transformers: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)

[8] DMIS Lab - BioBERT PyTorch: [https://github.com/dmis-lab/biobert-pytorch/tree/master/named-entity-recognition](https://github.com/dmis-lab/biobert-pytorch/tree/master/named-entity-recognition)
# BiLSTM + CRF for NER
Before running anything, make sure you have generated the pre-processed dataset using the generate_data.py file with the command mentioned in the parent directory. 

## Additional Requirements
1.	Packages: Anaconda, Pytorch, AllenNLP (if on linux and using elmo)  
2.	Glove 300B embeddings (If not using Elmo) 

## Training
1. Change settings in model/config.py  
2. Main settings to change: File directories, model hyperparameters etc.  
3. Run build_data.py  
    - Builds embedding dictionary, text file of words, chars tags, as well as idx to word and idx to char mapping for the model to read  
        
4. Run train.py  

## Results
#### Augmented with ADE Corpus Sample
|              |precision|    recall|  f1-score|   support|
|:---:|:---:|:---:|:---:|:---:|
|         ADE |      0.85 |     0.78 |     0.81  |    2219|
|         Dosage |      0.93 |     0.83 |     0.88  |     869|
|        Drug |      0.90 |     0.88 |     0.89  |    5037|
|         Duration |      0.77 |     0.57 |     0.65  |     130|
|         Form |      0.95 |     0.90 |     0.93  |    1384|
|         Frequency |      0.83 |     0.81 |     0.82  |    1220|
|         Reason |      0.70 |     0.26 |     0.38  |     743|
|         Route |      0.96 |     0.87 |     0.91  |    1079|
|         Strength |      0.92 |     0.92 |     0.92  |    1267|
|   micro avg |      0.89 |     0.82 |     0.86  |   13948|
|   macro avg |      0.87 |     0.76 |     0.80  |   13948|
|weighted avg |      0.89 |     0.82 |     0.85  |   13948|

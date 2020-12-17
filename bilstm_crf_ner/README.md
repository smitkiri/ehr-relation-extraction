# BiLSTM + CRF for NER
Before running anything, make sure you have generated the pre-processed dataset using the generate_data.py file with the command mentioned in the parent directory. 

## Additional Requirements
1.	Packages: Anaconda, Pytorch, AllenNLP  
2.	Glove 300B embeddings

## Training
1. Change settings in model/config.py  
2. Main settings to change: File directories, model hyperparameters etc.  
3. Run build_data.py  
    - Builds embedding dictionary, text file of words, chars tags, as well as idx to word and idx to char mapping for the model to read  
        
4. Run train.py  

## Results
|              |precision|    recall|  f1-score|
|:---:|:---:|:---:|:---:|
|         ADE |      0.1807 |     0.7168 |     0.2887  |
|         Dosage |      0.9272 |     0.9123 |     0.9197  |
|        Drug |      0.8898 |     0.9287|     0.9088  |
|         Duration |      0.8882 |     0.7778 |     0.8293  |
|         Form |      0.9840 |     0.9172 |     0.9494  |
|         Frequency |      0.9412 |     0.9494 |     0.9453  |
|         Reason |      0.7883 |     0.5238 |     0.6294  |
|         Route |      0.9583 |     0.9226 |     0.9401  |
|         Strength |      0.9769 |     0.9683 |     0.9726  |
|   micro avg |      0.8708 |     0.8957 |     0.8831  |
|   macro avg |      0.8624 |     0.8792 |     0.8684  |

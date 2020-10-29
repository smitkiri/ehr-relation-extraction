import argparse
from utils import read_data, generate_input_files
from typing import List
import warnings
import os

parser = argparse.ArgumentParser()   
parser.add_argument("--input_dir", type = str, 
                    help = "Directory with txt and ann files", 
                    default = "data/")

parser.add_argument("--target_dir", type = str, 
                    help = "Directory to save files", 
                    default = 'dataset/')

parser.add_argument("--max_seq_len", type = int, 
                    help = "Maximum sequence length", 
                    default = 510)

parser.add_argument("--dev_split", type = float, 
                    help = "Ratio of dev data", 
                    default = 0.1)

parser.add_argument("--test_split", type = float, 
                    help = "Ratio of test data", 
                    default = 0.2)

parser.add_argument("--tokenizer", type = str,
                    help = "The tokenizer to use. scispacy or default", 
                    default = "scispacy")

args = parser.parse_args()

labels = ['B-DRUG', 'I-DRUG' 'B-STR', 'I-STR', 'B-DUR', 'I-DUR' 
          'B-ROU', 'I-ROU', 'B-FOR', 'I-FOR', 'B-ADE', 'I-ADE',
          'B-DOS', 'I-DOS', 'B-REA', 'I-REA', 'B-FRE', 'I-FRE', 'O']

class Token:
    def __init__(self, text):
        self.text = text

def default_tokenizer(sequence: str) -> List[str]:
    """A tokenizer that splits sequence by a space."""
    words = sequence.split(' ')
    tokens = []
    for text in words:
        if not text:
            continue
        
        tokens.append(Token(text))
        
    return tokens

def main():
    
    if args.target_dir[-1] != '/':
        args.target_dir += '/'
        
    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)
    
    if args.tokenizer == "default":
        tokenizer = default_tokenizer
    
    elif args.tokenizer == "scispacy":
        import en_ner_bc5cdr_md
        tokenizer = en_ner_bc5cdr_md.load().tokenizer
    
    else:
        warnings.warn("Tokenizer named " + args.tokenizer + " not found."
                      "Using default tokenizer instead. Acceptable values"
                      "include 'scispacy' and 'default'.")
        tokenizer = default_tokenizer
    
    train_dev, test = read_data(data_dir = args.input_dir, 
                            train_ratio = 1 - args.test_split, 
                            tokenizer = tokenizer, verbose = 0)
    
    # Data is already shuffled, just split for dev set
    dev_split_idx = int((1 - args.dev_split) * len(train_dev))
    train = train_dev[:dev_split_idx]
    devel = train_dev[dev_split_idx:]
    
    files = {'train.txt': train, 'train_dev.txt': train_dev, 
             'devel.txt': devel, 'test.txt': test}
    
    # Generate train, dev, test files
    for filename, data in files.items():
        generate_input_files(ehr_records = data, 
                         filename = args.target_dir + filename, 
                         max_len = args.max_seq_len)
    
    # Generate labels file
    with open(args.target_dir + 'labels.txt', 'w') as file:
        output_labels = map(lambda x: x + '\n', labels)
        file.writelines(output_labels)
    
    print("\nGenerating files successful. Files generated: ", 
          ', '.join(list(files.keys())), ', labels.txt', sep = '')
    
if __name__ == '__main__':
    main()
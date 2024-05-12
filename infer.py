import argparse
import torch
import json
from torch.utils.data import DataLoader
from lstm1 import make_model_1, translate_beam, Seq2Seq, Encoder, Decoder
# from lstm2  import make_model_2, Seq2Seq, Encoder, Decoder, translate_beam2 #uncomment to run
from torch import Tensor
# from torchtext.vocab import GloVe
from typing import Tuple, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from torchtext.vocab import GloVe
# # import torchtext
glove_path = "glove"
glove = GloVe(name='6B', dim=100, cache=glove_path)
# # # Load pre-trained GloVe vectors (specify the dimensions, e.g., 100, 200, etc.)
# glove = GloVe(name='6B', dim=100)
import re


def tokenize_input(input_string):
    # Use regular expression to find numbers surrounded by words and separated by spaces
    matches = re.findall(r'(\d{1,3}(\s*,\s*\d{3})+)', input_string)
    
    # If matches are found, concatenate the numbers after removing commas
    if matches:
        for match in matches:
            number = match[0]
            # Remove commas and concatenate numbers
            concatenated_number = ''.join(number.split(","))
            # Replace the original match with the concatenated number
            input_string = input_string.replace(match[0], concatenated_number)
    return input_string


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def isnum(num):
    if num.isnumeric() or isfloat(num):
        return True
    else:
        return False

with open('archive/train.json', 'r') as f:
    training_data = json.load(f)

def get_vocab():
    unique_tokens_ip = set()
    unique_tokens_op = set()
    for ex in training_data:
        ex['Problem'] = tokenize_input(ex['Problem'])
        unique_tokens_ip.update(ex['Problem'].split(' '))
        unique_tokens_op.update(ex['linear_formula'].split('|'))

    input_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3, '<CONST>' : 4}  # Start with special tokens
    index = 5  # Start indexing from 4
    for token in sorted(unique_tokens_ip):
        if(isnum(token)):
            continue
        input_vocab[token] = index
        index += 1
    output_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3, '<CONST>' : 4}  # Start with special tokens
    index = 5  # Start indexing from 4
    for token in sorted(unique_tokens_op):
        if(isnum(token)):
            continue
        output_vocab[token] = index
        index += 1
    return input_vocab, output_vocab

input_vocab, output_vocab = get_vocab()
output_key_list = list(output_vocab.keys())
output_val_list = list(output_vocab.values())

def get_key_output(val):
    position = output_val_list.index(val)
    return output_key_list[position]
#############

input_key_list = list(input_vocab.keys())
input_val_list = list(input_vocab.values())

def get_key_input(val):
    position = input_val_list.index(val)
    return input_key_list[position]


class MathWordProblemDataset(Dataset):
    def __init__(self, file_path, input_vocab, output_vocab):
        self.file_path = file_path
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.training_data = self.load_data()
        self.preprocessed_data = self.preprocess_data()

    def load_data(self):
        # Load training data from JSON file
        with open(self.file_path, 'r') as f:
            return json.load(f)

    def tokenize_input(self, input_string):
        # Tokenize input string
        # You can use your tokenizer here if needed
        matches = re.findall(r'(\d{1,3}(\s*,\s*\d{3})+)', input_string)
        if matches:
            for match in matches:
                number = match[0]
                concatenated_number = ''.join(number.split(","))
                input_string = input_string.replace(match[0], concatenated_number)
        return input_string.split()

    def preprocess_data(self):
        preprocessed_data = []
        for example in self.training_data:
            input_tokens = self.tokenize_input(example['Problem'])
            input_tokens = ['<SOS>'] + input_tokens + ['<EOS>']
            input_tokens = ['<CONST>' if isnum(token) else token for token in input_tokens]
            input_indices = [self.input_vocab[token] if token in self.input_vocab else self.input_vocab['<UNK>'] for token in input_tokens]
            preprocessed_data.append((torch.tensor(input_indices)))
        return preprocessed_data


    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        input_indices = self.preprocessed_data[idx]
        return input_indices
    
    def collate_fn(self, batch):
        inputs = batch
        max_input_length = max(input_tensor.size(0) for input_tensor in inputs)
        padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        return padded_inputs

input_vocab_size = len(input_vocab)
output_vocab_size = len(output_vocab)
input_dim = input_vocab_size
output_dim = output_vocab_size
embedding_dim = 100  
hidden_dim = 256 
enc_hid_dim = 256
dec_hid_dim = 512 #double of encdim
dropout = 0.5  

def get_embedding(token):
    if token in glove.stoi:
        return glove.vectors[glove.stoi[token]]
    else:
        return torch.randn(100)

embedding_dim = 100  

# Initialize embedding tensor for input vocabulary
input_embedding_matrix = []

# Create embedding tensor for input vocabulary
for token in input_vocab:
    embedding = get_embedding(token)
    input_embedding_matrix.append(embedding)

# Convert input_embedding_matrix to a tensor
embedding_matrix = torch.stack(input_embedding_matrix)

def main():
    parser = argparse.ArgumentParser(description='Run inference on trained models')
    parser.add_argument('-model_file', type=str, help='Path to the trained model')
    parser.add_argument('--beam_size', type=int, choices=[1, 10, 20], help='Beam size')
    parser.add_argument('--model_type', type=str, choices=['lstm_lstm', 'lstm_lstm_attn', 'bert_lstm_attn_frozen', 'bert_lstm_attn_tuned'],
                        help='Model type')
    parser.add_argument('--test_data_file', type=str, help='Path to the JSON file containing the problems')

    args = parser.parse_args()

    device = torch.device('cpu')
    if args.model_type == 'lstm_lstm':
        weights_path = args.model_file
        model = make_model_1(input_vocab_size, output_vocab_size, weights_path, embedding_matrix)
        print(model)
    elif args.model_type == 'lstm_lstm_attn':
        weights_path = args.model_file
        model = make_model_2(input_vocab_size, output_vocab_size, weights_path, embedding_matrix)
        

    print(args.test_data_file)
    test_dataset = MathWordProblemDataset(str(args.test_data_file), input_vocab=input_vocab, output_vocab=output_vocab)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=test_dataset.collate_fn)

    # model = Seq2Seq.load_model(args.model_file, device)
    model.eval()
    if args.model_type == 'lstm_lstm':
        translated_sen_test = translate_beam(model, test_loader, output_vocab, get_key_output, device)
    else:
        translated_sen_test = translate_beam2(model, test_loader, device, 10, get_key_output, output_vocab)

    with open(args.test_data_file, 'r') as f:
        test_data = json.load(f)

    for i in range(len(test_data)):
        idx = 0
        while idx < len(translated_sen_test[i]):
            if translated_sen_test[i][idx] == "<EOS>":
                break
            idx += 1
        test_data[i]["predicted"] = '|'.join(translated_sen_test[i][1:idx])
    
    output_file = args.test_data_file.split('.')[0] + '_predicted.json'
    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=4)

    print("Inference completed. Predictions saved to", output_file)

if __name__ == "__main__":
    main()

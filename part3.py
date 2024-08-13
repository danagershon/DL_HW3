import unittest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.optim as optim
from tqdm import tqdm
import os


test = unittest.TestCase()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device:', device)

from hw3.transformer import sliding_window_attention

## test sliding-window attention
num_heads = 3
batch_size = 2
seq_len = 5
embed_dim = 3
window_size = 2

## test without extra dimension for heads
x = torch.arange(seq_len*embed_dim).reshape(seq_len,embed_dim).repeat(batch_size,1).reshape(batch_size, seq_len, -1).float()

values, attention = sliding_window_attention(x, x, x,window_size)

gt_values = torch.load(os.path.join('test_tensors','values_tensor_0_heads.pt'))

#print("me", values)
#print("desired", gt_values)

test.assertTrue(torch.all(values == gt_values), f'the tensors differ in dims [B,row,col]:{torch.stack(torch.where(values != gt_values),dim=0)}')

gt_attention = torch.load(os.path.join('test_tensors','attention_tensor_0_heads.pt'))
test.assertTrue(torch.all(attention == gt_attention), f'the tensors differ in dims [B,row,col]:{torch.stack(torch.where(attention != gt_attention),dim=0)}')


## test with extra dimension for heads
x = torch.arange(seq_len*embed_dim).reshape(seq_len,embed_dim).repeat(batch_size, num_heads, 1).reshape(batch_size, num_heads, seq_len, -1).float()

values, attention = sliding_window_attention(x, x, x,window_size)

gt_values = torch.load(os.path.join('test_tensors','values_tensor_3_heads.pt'))
test.assertTrue(torch.all(values == gt_values), f'the tensors differ in dims [B,num_heads,row,col]:{torch.stack(torch.where(values != gt_values),dim=0)}')


gt_attention = torch.load(os.path.join('test_tensors','attention_tensor_3_heads.pt'))
test.assertTrue(torch.all(attention == gt_attention), f'the tensors differ in dims [B,num_heads,row,col]:{torch.stack(torch.where(attention != gt_attention),dim=0)}')
print("passed")

import numpy as np
import pandas as pd
import sys
import pathlib
import urllib
import shutil
import re

import matplotlib.pyplot as plt


from datasets import DatasetDict
from datasets import load_dataset, load_metric, concatenate_datasets

dataset = load_dataset('imdb', split=['train', 'test', 'train[12480:12520]'])

print(dataset)

#wrap it in a DatasetDict to enable methods such as map and format
dataset = DatasetDict({'train': dataset[0], 'val': dataset[1], 'toy': dataset[2]})

dataset

print(dataset['train'])

for i in range(4):
    print(f'TRAINING SAMPLE {i}:') 
    print(dataset['train'][i]['text'])
    label = dataset['train'][i]['label']
    print(f'Label {i}: {label}')
    print('\n')

def label_cnt(type):
    ds = dataset[type]
    size = len(ds)
    cnt= 0 
    for smp in ds:
        cnt += smp['label']
    print(f'negative samples in {type} dataset: {size - cnt}')
    print(f'positive samples in {type} dataset: {cnt}')
    
label_cnt('train')
label_cnt('val')
label_cnt('toy')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer input max length:", tokenizer.model_max_length)
print("Tokenizer vocabulary size:", tokenizer.vocab_size)

def tokenize_text(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

def tokenize_dataset(dataset):
    dataset_tokenized = dataset.map(tokenize_text, batched=True, batch_size =None)
    return dataset_tokenized

dataset_tokenized = tokenize_dataset(dataset)

# we would like to work with pytorch so we can manually fine-tune
dataset_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# no need to parrarelize in this assignment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.utils.data import DataLoader, Dataset

class IMDBDataset(Dataset):
    def __init__(self, dataset):
        self.ds = dataset

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return self.ds.num_rows

train_dataset = IMDBDataset(dataset_tokenized['train'])
val_dataset = IMDBDataset(dataset_tokenized['val'])
toy_dataset = IMDBDataset(dataset_tokenized['toy'])

dl_train,dl_val, dl_toy = [ 
    DataLoader(
    dataset=train_dataset,
    batch_size=12,
    shuffle=True, 
    num_workers=0
),
DataLoader(
    dataset=val_dataset,
    batch_size=12,
    shuffle=True, 
    num_workers=0
),
DataLoader(
    dataset=toy_dataset,
    batch_size=4,
    num_workers=0
)]

from hw3.transformer import EncoderLayer
# set torch seed for reproducibility
torch.manual_seed(0)
layer = EncoderLayer(embed_dim=16, hidden_dim=16, num_heads=4, window_size=4, dropout=0.1)

# load x and y
x = torch.load(os.path.join('test_tensors','encoder_layer_input.pt'))
y = torch.load(os.path.join('test_tensors','encoder_layer_output.pt'))
padding_mask = torch.ones(2, 10)
padding_mask[:, 5:] = 0
#print(x.shape)
# forward pass
out = layer(x, padding_mask)
#print("me",out)
#print("desired", y)
test.assertTrue(torch.allclose(out, y, atol=1e-6), 'output of encoder layer is incorrect')


tokenizer.convert_ids_to_tokens(dataset_tokenized['train'][0]['input_ids'])[:10]

from hw3.transformer import Encoder

# set torch seed for reproducibility
torch.manual_seed(0)
encoder = Encoder(vocab_size=100, embed_dim=16, num_heads=4, num_layers=3, 
                  hidden_dim=16, max_seq_length=64, window_size=4, dropout=0.1)


# load x and y
x = torch.load(os.path.join('test_tensors','encoder_input.pt'))
y = torch.load(os.path.join('test_tensors','encoder_output.pt'))
#x = torch.randint(0, 100, (2, 64)).long()

padding_mask = torch.ones(2, 64)
padding_mask[:, 50:] = 0

# forward pass
out = encoder(x, padding_mask)
test.assertTrue(torch.allclose(out, y, atol=1e-6), 'output of encoder layer is incorrect')


from hw3.answers import part3_transformer_encoder_hyperparams

params = part3_transformer_encoder_hyperparams()
print(params)
embed_dim = params['embed_dim'] 
num_heads = params['num_heads']
num_layers = params['num_layers']
hidden_dim = params['hidden_dim']
window_size = params['window_size']
dropout = params['dropout']
lr = params['lr']

vocab_size = tokenizer.vocab_size
max_seq_length = tokenizer.model_max_length

max_batches_per_epoch = None
N_EPOCHS = 20

toy_model = Encoder(vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout=dropout).to(device)
toy_optimizer = optim.Adam(toy_model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

# fit your model
import pickle
TRAIN = False

if not os.path.exists('toy_transfomer_encoder.pt') or TRAIN:
    # overfit
    from hw3.training import TransformerEncoderTrainer
    toy_trainer = TransformerEncoderTrainer(toy_model, criterion, toy_optimizer, device)
    # set max batches per epoch
    _ = toy_trainer.fit(dl_toy, dl_toy, N_EPOCHS, checkpoints='toy_transfomer_encoder', max_batches=max_batches_per_epoch)

toy_saved_state = torch.load('toy_transfomer_encoder.pt', map_location=device)
toy_best_acc = toy_saved_state['best_acc']
toy_model.load_state_dict(toy_saved_state['model_state']) 
print(f"toy_best_acc: {toy_best_acc}")

test.assertTrue(toy_best_acc >= 95)

max_batches_per_epoch = 500
N_EPOCHS = 4

model = Encoder(vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# fit your model
import pickle

if not os.path.exists('trained_transfomer_encoder.pt'):
    from hw3.training import TransformerEncoderTrainer
    trainer = TransformerEncoderTrainer(model, criterion, optimizer, device=device)
    # set max batches per epoch
    _ = trainer.fit(dl_train, dl_val, N_EPOCHS, checkpoints='trained_transfomer_encoder', max_batches=max_batches_per_epoch)

saved_state = torch.load('trained_transfomer_encoder.pt', map_location=device)
best_acc = saved_state['best_acc']
model.load_state_dict(saved_state['model_state']) 
print(f"best_acc: {best_acc}")

test.assertTrue(best_acc >= 65)

rand_index = torch.randint(len(dataset_tokenized['val']), (1,))
rand_index

sample = dataset['val'][rand_index]
sample['text']

model.to(device)
tokenized_sample = dataset_tokenized['val'][rand_index]
tokenized_sample
input_ids = tokenized_sample['input_ids'].to(device)
label = tokenized_sample['label'].to(device)
attention_mask = tokenized_sample['attention_mask'].to(float).to(device)

print('label', label.shape)
print('attention_mask', attention_mask.shape)
prediction = model.predict(input_ids, attention_mask).squeeze(0)

print('label: {}, prediction: {}'.format(label, prediction))

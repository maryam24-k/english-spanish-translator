#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import sys
import os
sys.path.append(os.path.abspath(f'{os.getenv("HOME")}/Dropbox/is/ai/common'))
from rui.utils import TextVectorizer
from rui.torch.utils import train, evaluate, plotEpoch, ModelCheckpoint
from rui.torch.transformer import PositionalEmbedding, Encoder
 
from pathlib import Path
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
 
gpu_id = 0 # Change to your desired GPU index (e.g., 0, 1, 2)
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
 
vocab_size = 20000
 
os.chdir(f'{os.getenv("HOME")}/Data')  # https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
 
 
#%% load imdb data
 
def load_imdb_data(data_dir):    
    texts, labels = [], []
    data_path = Path(data_dir)
    for label_type in ['pos', 'neg']:
        dir_path = data_path / label_type
        label = 1 if label_type == 'pos' else 0
        if not dir_path.exists():
            print(f"Warning: Directory {dir_path} does not exist")
            continue
        for file_path in dir_path.glob('*.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append( f.read() )
                labels.append( label )
    print(f"Loaded {len(texts)} samples from {data_dir}")
    return texts, labels
 
 
train_texts, train_labels = load_imdb_data("imdb/train")
val_texts, val_labels = load_imdb_data("imdb/val")
test_texts, test_labels = load_imdb_data("imdb/test")
 
#%%
 
class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer=None, precomputed_features=None):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.vectorizer = vectorizer
        self.precomputed_features = precomputed_features
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        if self.precomputed_features is not None:
            return torch.from_numpy(self.precomputed_features[i]), self.labels[i]
        if self.vectorizer:
            features = torch.from_numpy( self.vectorizer( [self.texts[i]] )[0] )
            return features, self.labels[i]
        return self.texts[i], self.labels[i]
 
def create_dataloaders(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vectorizer, batch_size=32, precompute=True):    
    if precompute:
        print("Precomputing features...")
        train_features = vectorizer(train_texts)
        val_features = vectorizer(val_texts)
        test_features = vectorizer(test_texts)
        train_ds = TextDataset(train_texts, train_labels, precomputed_features=train_features)
        val_ds = TextDataset(val_texts, val_labels, precomputed_features=val_features)
        test_ds = TextDataset(test_texts, test_labels, precomputed_features=test_features)
    else:
        train_ds = TextDataset(train_texts, train_labels, vectorizer=vectorizer)
        val_ds = TextDataset(val_texts, val_labels, vectorizer=vectorizer)
        test_ds = TextDataset(test_texts, test_labels, vectorizer=vectorizer)
    # Use num_workers=0 if dataset is small or precomputed to avoid overhead
    num_workers = 0 if precompute else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    return train_loader, val_loader, test_loader
 
 
#%% n-gram model
 
class NGramModel(nn.Module):
    def __init__(self, max_tokens=20000, hidden_dim=16, num_classes=2):
        super().__init__()
        self.max_tokens = max_tokens
        self.fc1 = nn.Linear(max_tokens, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
 
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
 
batch_size = 32
    
#%% Unigram with multi-hot encoding
 
vectorizer = TextVectorizer( max_tokens=vocab_size, output_mode="multi_hot" )
vectorizer.adapt( train_texts )
train_loader, val_loader, test_loader = create_dataloaders( train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vectorizer, batch_size=batch_size )
 
I = iter(test_loader.dataset)
x,y = next(I)
x.size()
 
model = NGramModel( max_tokens=vocab_size )
optimizer = torch.optim.RMSprop(model.parameters())
callbacks = [ ModelCheckpoint(filepath="models/imdb_1gram.pt") ]
history = train( model, train_loader, val_loader, optimizer, device=device, n_batch_per_report=100, n_epochs=5, callbacks=callbacks ) 
plotEpoch(history, metric="accuracy")
 
test_result = evaluate(model, test_loader, device) 
print(f"Test Loss: {test_result['loss']:.2f}\nTest ACC: {test_result['accuracy']:.2f}") # 88%
 
#%% Bigram with multi-hot encoding
 
vectorizer = TextVectorizer( max_tokens=vocab_size, output_mode="multi_hot", ngrams=2 )
vectorizer.adapt( train_texts )
train_loader, val_loader, test_loader = create_dataloaders( train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vectorizer, batch_size=batch_size )
 
model = NGramModel( max_tokens=vocab_size )
optimizer = torch.optim.RMSprop(model.parameters())
callbacks = [ ModelCheckpoint(filepath="models/imdb_2gram.pt") ]
history = train(model, train_loader, val_loader, optimizer, callbacks=callbacks, device=device, n_batch_per_report=100, n_epochs=5)
plotEpoch(history, metric="accuracy")
 
test_result = evaluate(model, test_loader, device) 
print(f"Test Loss: {test_result['loss']:.2f}\nTest ACC: {test_result['accuracy']:.2f}") # 89%
 
#%% Bigram with TF-IDF
 
vectorizer = TextVectorizer(max_tokens=vocab_size, output_mode="tf_idf", ngrams=2)
vectorizer.adapt( train_texts )
train_loader, val_loader, test_loader = create_dataloaders( train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vectorizer, batch_size=batch_size )
 
model = NGramModel( max_tokens=vocab_size )
optimizer = torch.optim.RMSprop( model.parameters() )
callbacks = [ ModelCheckpoint(filepath="models/imdb_tfidf.pt") ]
history = train(model, train_loader, val_loader, optimizer, callbacks=callbacks, device=device, n_batch_per_report=100, n_epochs=5)
plotEpoch(history, metric="accuracy")
 
test_result = evaluate(model, test_loader, device) 
print(f"Test Loss: {test_result['loss']:.2f}\nTest ACC: {test_result['accuracy']:.2f}") # 87%
 
#%% Inference
 
samples = ["The movie is not particularly interesting! But the acting is superb", "this movie sucks!"]
features = torch.from_numpy(vectorizer( samples )).to(device) # ensure the SAME vectorizer is used for inference
with torch.no_grad():
    pred = model( features )
F.softmax(pred, dim=-1)
 
#%% Data Preparation for Sequence Models
"""
RNN and Transformer ingests each instance as a sequence
Padding: shorter sequences are padded with 0 at the end so that they can be concatenated with other sequences to form contiguous batches
Masking: to prevent RNN from spending the last iterations processing meaningless paddings which results in the fading of useful information stored in the internal states
"""
seq_len = 600  # truncate the sequence after the first 600 tokens. Average review length is 233. Only 5% of reviews are longer than 600
vectorizer_int = TextVectorizer(max_tokens=vocab_size, output_sequence_length=seq_len, output_mode="int")
vectorizer_int.adapt( train_texts )
train_loader_seq, val_loader_seq, test_loader_seq = create_dataloaders( train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, vectorizer_int, batch_size=batch_size )
 
#%%
 
class BiGRUModel(nn.Module):    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=16, num_classes=2):
        super().__init__()
        # Embedding() takes a rank-2 tensor shaped (batch, seq_len) and returns a rank-3 tensor shaped (batch, seq_len, embedding_dimension)
        self.embedding = nn.Embedding( vocab_size, embedding_dim, padding_idx=0 )
        self.gru = nn.GRU( embedding_dim, hidden_dim, batch_first=True, bidirectional=True )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear( hidden_dim * 2, num_classes )  # *2 for bidirectional
 
    def forward(self, x):
        # Create mask for packed sequence (optional, GRU handles padding reasonably)
        lengths = (x != 0).sum(dim=1).cpu()
        lengths = lengths.clamp(min=1)  # Ensure minimum length of 1
        x = self.embedding(x)
        # Pack padded sequence for efficient computation
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)
 
model = BiGRUModel(vocab_size).to(device)
optimizer = torch.optim.RMSprop( model.parameters() )
callbacks = [ ModelCheckpoint(filepath="models/imdb_gru.pt") ]
history = train(model, train_loader_seq, val_loader_seq, optimizer, callbacks=callbacks, device=device, n_batch_per_report=100, n_epochs=5 )
plotEpoch(history, metric="accuracy") # 85.56%
 
test_result = evaluate(model, test_loader_seq, device) 
print(f"Test Loss: {test_result['loss']:.2f}\nTest ACC: {test_result['accuracy']:.2f}") # 85%
 
#%% Transformer
 
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_emb, n_layers=2, n_heads=4, d_ff=32, dropout=0.5, num_classes=2, prenorm=False):
        super().__init__()
        self.pos_embedding = PositionalEmbedding(vocab_size, d_emb)
        self.encoder = Encoder(n_layers=n_layers, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, dropout=0.1, prenorm=prenorm)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_emb, num_classes)
 
    def forward(self, x):
        # Create padding mask for attention (True for positions to mask)
        key_padding_mask = (x == 0)
        x = self.pos_embedding(x)
        x = self.encoder(x)
        # Global max pooling: (batch, seq_len, d_emb) -> (batch, d_emb)
        # Mask padded positions before max pooling
        mask = key_padding_mask.unsqueeze(-1).expand_as(x)
        x = x.masked_fill(mask, float('-inf'))
        x, _ = x.max(dim=1)  # max pooling over sequence
        x = self.dropout(x)
        return self.fc(x)
 
model = TransformerClassifier(vocab_size, d_emb=128, n_layers=1, n_heads=1, d_ff=32, prenorm=True)
optimizer = torch.optim.RMSprop( model.parameters(), lr=1e-3 )
callbacks = [ ModelCheckpoint(filepath="models/imdb_transformer.pt") ]
history = train( model, train_loader_seq, val_loader_seq, optimizer, device=device, n_batch_per_report=100, n_epochs=4)#, callbacks=callbacks )
plotEpoch(history, metric="accuracy") # ~ 88% in 4 epochs
 
test_result = evaluate(model, test_loader_seq, device)
print(f"Test Loss: {test_result['loss']:.2f}\nTest ACC: {test_result['accuracy']:.2f}") # 88%
 
#%% BERT
 
from rui.torch.bert import BertClassifier, BertPreprocessor
 
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
 
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
 
train_inputs, train_masks = BertPreprocessor( train_texts ) # both are torch.Tensor
train_labels = torch.tensor(train_labels, dtype=torch.long) # convert to torch.Tensor
train_data = TensorDataset(train_inputs, train_masks, train_labels) # wrap 3 tensors
train_sampler = RandomSampler(train_data)
train_loader_bert = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) # the DataLoader constructor takes a DataSet object as input, along with batch_size and other options
 
val_inputs, val_masks = BertPreprocessor( val_texts )
val_labels = torch.tensor(val_labels, dtype=torch.long)
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler( val_data )
val_loader_bert = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
 
test_inputs, test_masks = BertPreprocessor( test_texts )
test_labels = torch.tensor( test_labels, dtype=torch.long )
test_data = TensorDataset( test_inputs, test_masks, test_labels )
test_sampler = SequentialSampler( test_data )
test_loader_bert = DataLoader( test_data, sampler=test_sampler, batch_size=batch_size )
 
model = BertClassifier(freeze_bert=False)
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8 )
callbacks = [ModelCheckpoint(filepath="models/imdb_bert.pt", save_best_only=True, monitor="val_loss")]
 
n_epochs = 4
scheduler = get_linear_schedule_with_warmup( optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * n_epochs ) # Set up the learning rate scheduler
history = train( model, train_loader_bert, val_loader_bert, optimizer, scheduler=scheduler, callbacks=callbacks, device=device, n_batch_per_report=100, n_epochs=n_epochs )
plotEpoch(history, metric="accuracy" ) # 88% at epoch 4
 
test_result = evaluate(model, test_loader_bert, device)
print(f"Test Loss: {test_result['loss']:.2f}\nTest ACC: {test_result['accuracy']:.2f}")# ~88% at epoch 3
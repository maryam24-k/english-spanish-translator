#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
 
#%%
 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
       
class BertClassifier( nn.Module ):
    def __init__(self, freeze_bert=False):
        super().__init__()
        config = BertConfig( max_position_embeddings=1024 ) # Instantiate BERT model, default hidden_size=767, i.e., dimensionality of the encoder layers and the pooler layer
        BertModel( config )
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        D_in, H, D_out = 768, 50, 2 # D_in is hidden_size of BERT, H is the hidden size of our classifier, and D_out is the number of classes
        self.classifier = nn.Sequential( nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, D_out) )  # Instantiate a one-layer feed-forward classifier.  nn.Dropout(0.5),
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False # if True, PyTorch will track all tensors that have param as an ancestor
 
    def forward(self, x):
        input_ids, attention_mask = x
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # calling an instance of nn.Module ends up calling forward() with the same arguments
        last_hidden_state_cls = outputs[0][:, 0, :] # Extract the last hidden state of the token `[CLS]` for classification task
        logits = self.classifier( last_hidden_state_cls )
        return logits
 
 
def BertPreprocessor( txt ):
    input_ids = []
    attention_masks = []
    for s in txt:
        s = re.sub(r'(@.*?)[\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        encoded = tokenizer.encode_plus( text=s, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_attention_mask=True )
        input_ids.append( encoded.get('input_ids') ) # Add the outputs to the lists
        attention_masks.append( encoded.get('attention_mask') )
    return torch.tensor( input_ids ), torch.tensor( attention_masks )

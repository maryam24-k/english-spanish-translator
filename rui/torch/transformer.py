#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
#%% Positional Embedding
 
def positional_encoding(seq_len, depth):
    d = depth / 2
    positions = np.arange(seq_len)[:, np.newaxis]  # shape: (seq_len, 1)
    d = np.arange(d)[np.newaxis, :] / d  # shape: (1, d)
    angle_rates = 1 / (10000 ** d)  # shape: (1, d)
    angle_rads = positions * angle_rates  # shape: (seq_len, d)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)  # shape: (seq_len, 2*d)
    return torch.tensor(pos_encoding, dtype=torch.float32)  # shape: (seq_len, depth)
 
class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_emb, padding_idx=0, seq_len=1024):
        super().__init__()
        self.d_emb = d_emb  # d_emb is the dimension of positional embedding, shared by the encoder and the decoder
        self.embedding = nn.Embedding(vocab_size, d_emb, padding_idx=padding_idx) # token embedding
        self.register_buffer('pos_encoding', positional_encoding(seq_len=seq_len, depth=d_emb))  # position embedding, with maximum sequence length 2048
        self._init_weights()
    def _init_weights(self): # uniform initialization for embeddings, as in Keras        
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05) # PyTorch methods ending with an underscore makes changes in-place
        self.embedding._fill_padding_idx_with_zero() # Keep padding as zeros
    def forward(self, x):
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= math.sqrt(self.d_emb)  # This factor sets the relative scale of the embedding and positonal encoding.
        return x + self.pos_encoding[:seq_len, :].unsqueeze(0)  # use the position embedding matrix up to the seq_len position (i.e., row)
 
#%% Attention and FeedForward
 
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, seq_len=1024, qkv_bias=False, is_causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.d_model = embed_dim
        self.n_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.is_causal = is_causal
        if self.is_causal: # fill 1 in upper triangular (excluding diagonal) part to indicate masking
            self.register_buffer('mask', torch.triu(torch.ones(seq_len, seq_len), diagonal=1))
        # register a tensor as part of the module's state, without it being a learnable parameter
        self._init_weights()
        
    def _init_weights(self):        
         for name, param in self.named_parameters():
             if 'weight' in name:
                 nn.init.xavier_uniform_(param) # Xavier/Glorot initialization (same as Keras default)
             elif 'bias' in name:
                 nn.init.zeros_(param)
 
    def forward(self, q, k, v):
        b, seq_len, d_in = q.shape
        k = self.W_k(k) # Shape: (b, seq_len, d_model)
        q = self.W_q(q)
        v = self.W_v(v)
        # Implicitly split the matrix by adding a `n_heads` dimension: (b, seq_len, d_model) -> (b, seq_len, n_heads, d_k)
        k = k.view(b, seq_len, self.n_heads, self.d_k) # (b, seq_len, n_heads, d_k)
        v = v.view(b, seq_len, self.n_heads, self.d_k) # (b, seq_len, n_heads, d_k)
        q = q.view(b, seq_len, self.n_heads, self.d_k) # (b, seq_len, n_heads, d_k)    
        k = k.transpose(1, 2) # (b, n_heads, seq_len, d_k)
        q = q.transpose(1, 2) # (b, n_heads, seq_len, d_k)
        v = v.transpose(1, 2) # (b, n_heads, seq_len, d_k)       
        attn_scores = q @ k.transpose(2, 3) # (b, n_heads, seq_len, seq_len)
        if self.is_causal:
            mask_bool = self.mask.bool()[:seq_len, :seq_len] # truncate the mask to seq_len and convert to boolean         
            attn_scores.masked_fill_(mask_bool, -torch.inf) # fill masked position with -torch.inf
        attn_weights = torch.softmax(attn_scores / self.d_k**0.5, dim=-1) # (b, n_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights) # (b, n_heads, seq_len, seq_len)
        context_vec = (attn_weights @ v).transpose(1, 2) # (b, n_heads, seq_len, d_k), then (b, seq_len, n_heads, d_k) after transpose
        context_vec = context_vec.contiguous().view(b, seq_len, self.d_model) # Combine heads: (b, seq_len, n_heads * d_k) where d_model = n_heads * d_k
        context_vec = self.out_proj(context_vec) # optional projection
        return context_vec
 
 
class FeedForward(nn.Module):
    def __init__(self, d_emb, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_emb, d_ff)
        self.linear2 = nn.Linear(d_ff, d_emb)
        self._init_weights()
    def _init_weights(self): # Xavier/Glorot initialization (same as Keras default)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    def forward(self, x):
        out = F.gelu(self.linear1(x)) # F.gelu or F.relu
        return self.linear2(out)
    
#%% Transformer = Encoder + Decoder + FeedForward
 
class EncoderLayer(nn.Module):
    def __init__(self, *, d_emb, n_heads, d_ff, dropout=0.1, prenorm=False): # default: post layernorm, as in the original paper
        super().__init__()
        self.gsa = MultiHeadAttention( embed_dim=d_emb, num_heads=n_heads, dropout=dropout )
        self.ff = FeedForward(d_emb, d_ff)
        self.norm1 = nn.LayerNorm( d_emb )
        self.norm2 = nn.LayerNorm( d_emb )
        self.drop_shortcut = nn.Dropout( dropout )
        self.prenorm = prenorm
    def forward(self, x):
        if self.prenorm:
            shortcut = x # Shortcut connection for attention block
            x = self.norm1(x)
        else:
            x = self.norm1(x)
            shortcut = x
        x = self.gsa(x,x,x)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back
        if self.prenorm:
            shortcut = x # Shortcut connection for feed forward block
            x = self.norm2(x)
        else:
            x = self.norm2(x)
            shortcut = x
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut  # Add the original input back
 
 
class Encoder(nn.Module):
    def __init__(self, *, n_layers, d_emb, n_heads, d_ff, dropout=0.1, prenorm=False):
        super().__init__()
        self.d_emb = d_emb
        self.n_layers = n_layers
        self.enc_layers = nn.ModuleList( [EncoderLayer(d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, prenorm=prenorm) for _ in range(n_layers)] )
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x):  # x is token embeddings shape: (batch, seq_len, d_emb)
        x = self.dropout(x)
        for i in range(self.n_layers):
            x = self.enc_layers[i](x)
        return x  # Shape: (batch_size, seq_len, d_emb)
 
 
class DecoderLayer(nn.Module):
    def __init__(self, *, d_emb, n_heads, d_ff, seq_len=1024, dropout=0.1, prenorm=False):
        super().__init__()
        self.csa = MultiHeadAttention( embed_dim=d_emb, num_heads=n_heads, dropout=dropout, seq_len=seq_len, is_causal=True)
        self.ca = MultiHeadAttention( embed_dim=d_emb, num_heads=n_heads, dropout=dropout )
        self.ff = FeedForward(d_emb, d_ff)
        self.norm1 = nn.LayerNorm( d_emb )
        self.norm2 = nn.LayerNorm( d_emb )
        self.drop_shortcut = nn.Dropout( dropout )
        self.prenorm = prenorm
        
    def forward(self, x, context):
        if self.prenorm:
            shortcut = x # Shortcut connection for attention block
            x = self.norm1(x)
        else:
            x = self.norm1(x)
            shortcut = x
        x = self.csa(x,x,x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.ca(q=x, k=context, v=context)
        x = x + shortcut  # Add the original input back
        if self.prenorm:
            shortcut = x # Shortcut connection for feed forward block
            x = self.norm2(x)
        else:
            x = self.norm2(x)
            shortcut = x
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut  # Add the original input back
    
 
class Decoder(nn.Module):
    def __init__(self, *, n_layers, d_emb, n_heads, d_ff, seq_len=1024, dropout=0.1, prenorm=False):
        super().__init__()
        self.d_emb = d_emb
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.dec_layers = nn.ModuleList( [DecoderLayer(d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, seq_len=seq_len, dropout=dropout, prenorm=prenorm) for _ in range(n_layers)] )
 
    def forward(self, x, context):  # x is of shape (batch, tgt_seq_len, d_emb); context is of shape (batch_size, context_len, d_emb)
        x = self.dropout(x)
        for i in range(self.n_layers):
            x = self.dec_layers[i](x, context)
        return x  # shape: (batch_size, tgt_seq_len, d_emb)
 
 
class Transformer(nn.Module):
    def __init__(self, *, n_layers, d_emb, n_heads, d_ff, src_vocab_size, tgt_vocab_size, seq_len=1024, dropout=0.1, prenorm=False ):
        super().__init__()
        self.src_pos_embedding = PositionalEmbedding(vocab_size=src_vocab_size, d_emb=d_emb, seq_len=seq_len)
        self.tgt_pos_embedding = PositionalEmbedding(vocab_size=tgt_vocab_size, d_emb=d_emb, seq_len=seq_len)
        self.encoder = Encoder(n_layers=n_layers, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, dropout=dropout, prenorm=prenorm)
        self.decoder = Decoder(n_layers=n_layers, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff, seq_len=seq_len, dropout=dropout, prenorm=prenorm)
        self.final_layer = nn.Linear(d_emb, tgt_vocab_size)
 
    def forward(self, x):
        src, tgt = x
        src_emb = self.src_pos_embedding(src)
        tgt_emb = self.tgt_pos_embedding(tgt)
        context = self.encoder(src_emb)  # (batch_size, context_len, d_emb)
        x = self.decoder(tgt_emb, context)  # (batch_size, target_len, d_emb)
        return self.final_layer(x)  # (batch_size, target_len, tgt_vocab_size)
 
 
#%% Sanity Check: run from the parent folder of rui as "python -m rui.torch.transformer" since we used relative import
if __name__ == "__main__":
    
    from ..utils import TextVectorizer
    
    seq_len = 10
    max_vocab_size = 100
    d_emb = 512
    
    vectorizer = TextVectorizer( max_tokens=max_vocab_size, output_mode="int" )
    txts = ['how are you']
    vectorizer.adapt( txts )
    encoded = vectorizer(txts)
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoded_tensor = torch.tensor(encoded).to(device)
 
    x = PositionalEmbedding(vocab_size=max_vocab_size, d_emb=d_emb, seq_len=seq_len).to(device)(encoded_tensor)
    
    n_heads = 8  
    mha = MultiHeadAttention( d_emb, n_heads).to(device)
    print(f"MultiHeadAttention output shape: {mha(x,x,x).shape}")
 
    d_ff = 2048
    encoder_layer = EncoderLayer(d_emb=d_emb, n_heads=n_heads, d_ff=d_ff).to(device)
    decoder_layer = DecoderLayer(d_emb=d_emb, n_heads=n_heads, d_ff=d_ff).to(device)
    print(f"EncoderLayer output shape: {encoder_layer(x).shape}")
 
    n_layers = 6
    encoder = Encoder(n_layers=n_layers, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff).to(device)
    decoder = Decoder(n_layers=n_layers, d_emb=d_emb, n_heads=n_heads, d_ff=d_ff).to(device)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        print(f"Encoder output shape: {encoder(x).shape}")
        print(f"Decoder output shape: {decoder(x=x, context=x).shape}")
else:
    print(f'Transformer imported from local file "{__name__}.py"')
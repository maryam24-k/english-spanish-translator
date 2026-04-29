#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import numpy as np
import torch
import torch.nn as nn
 
from .transformer import MultiHeadAttention, FeedForward  
 
GPT2CONFIG = {"vocab_size": 50257, "context_length": 1024, "drop_rate": 0.0, "qkv_bias": True }
GPT2SIZE = { "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "size": "124M"},  # 621.83 MB
             "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "size":"355M"}, # 1549.58 MB
             "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "size":"774M"},  # 3197.56 MB
             "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "size":"1558M"} }   # 6247.68 MB
   
class TransformerBlock(nn.Module): # Pre-LayerNorm, instead of Post-LayerNorm in the original paper
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention( embed_dim=cfg["emb_dim"], num_heads=cfg["n_heads"], seq_len=cfg["context_length"], dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"], is_causal=True )
        #self.ff = FeedForward( cfg )
        self.ff = FeedForward( cfg["emb_dim"], 4*cfg["emb_dim"] )
        self.norm1 = nn.LayerNorm( cfg["emb_dim"] )
        self.norm2 = nn.LayerNorm( cfg["emb_dim"] )
        self.drop_shortcut = nn.Dropout( cfg["drop_rate"] )
 
    def forward(self, x):        
        shortcut = x # Shortcut connection for attention block
        x = self.norm1(x)
        x = self.att(x,x,x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back        
        shortcut = x # Shortcut connection for feed forward block
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        return x
 
 
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb  = nn.Embedding( cfg["vocab_size"], cfg["emb_dim"] )
        self.pos_emb  = nn.Embedding( cfg["context_length"], cfg["emb_dim"] )
        self.drop_emb = nn.Dropout( cfg["drop_rate"] )
        self.trf_blocks = nn.Sequential( *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])] )       
        self.final_norm = nn.LayerNorm( cfg["emb_dim"] )
        self.out_head   = nn.Linear( cfg["emb_dim"], cfg["vocab_size"], bias=False )
 
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
 
 
#%% utility functions for training and inference
 
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor
 
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
 
# generate based on idx, context of shape (batch, seq_len) with each element a token ID
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):    
    for _ in range(max_new_tokens): 
        idx_cond = idx[:, -context_size:] # crop current context to at most the last context_size number of token ids
        with torch.no_grad():
            logits = model(idx_cond) # (batch, seq_len, vocab_size)
        logits = logits[:, -1, :] # By focusing on the last time step, the shape becomes (batch, vocab_size)
        if top_k is not None: # keep only top_k values
            top_logits, _ = torch.topk(logits, top_k) # returns top_k logits in descending order and their positions
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0.0: # apply temperature scaling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # logits.shape = (batch_size, vocab_size)            
            idx_next = torch.multinomial(probs, num_samples=1) # idx_next.shape = (batch_size, 1)
        else: # greedy selection of the idx of the vocab with the highest probability
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # idx_next.shape = (batch_size, 1)
        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break
        idx = torch.cat((idx, idx_next), dim=1) # idx.shape = (batch_size, seq_len+1) after appending id_next to idx
    return idx
 
#%% utility functions for loading pretrained GPT2 models
 
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
 
def load_pretrained(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split( (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1 )
        gpt.trf_blocks[b].att.W_q.weight = assign( gpt.trf_blocks[b].att.W_q.weight, q_w.T )
        gpt.trf_blocks[b].att.W_k.weight = assign( gpt.trf_blocks[b].att.W_k.weight, k_w.T )
        gpt.trf_blocks[b].att.W_v.weight = assign( gpt.trf_blocks[b].att.W_v.weight, v_w.T )
        # load bias
        q_b, k_b, v_b = np.split( (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1 )
        gpt.trf_blocks[b].att.W_q.bias = assign( gpt.trf_blocks[b].att.W_q.bias, q_b )
        gpt.trf_blocks[b].att.W_k.bias = assign( gpt.trf_blocks[b].att.W_k.bias, k_b )
        gpt.trf_blocks[b].att.W_v.bias = assign( gpt.trf_blocks[b].att.W_v.bias, v_b )
        # load weights and bias for the output projection layer of mha
        gpt.trf_blocks[b].att.out_proj.weight = assign( gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T )
        gpt.trf_blocks[b].att.out_proj.bias = assign(   gpt.trf_blocks[b].att.out_proj.bias,   params["blocks"][b]["attn"]["c_proj"]["b"] )
        # load weights and bias for feedforward layers
        gpt.trf_blocks[b].ff.layers[0].weight = assign( gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(   gpt.trf_blocks[b].ff.layers[0].bias,   params["blocks"][b]["mlp"]["c_fc"]["b"] )
        gpt.trf_blocks[b].ff.layers[2].weight = assign( gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(   gpt.trf_blocks[b].ff.layers[2].bias,   params["blocks"][b]["mlp"]["c_proj"]["b"] )
        # load parameters for scale and shift of LayerNorm layers
        gpt.trf_blocks[b].norm1.scale = assign( gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"] )
        gpt.trf_blocks[b].norm1.shift = assign( gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"] )
        gpt.trf_blocks[b].norm2.scale = assign( gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"] )
        gpt.trf_blocks[b].norm2.shift = assign( gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"] )
    gpt.final_norm.scale = assign( gpt.final_norm.scale, params["g"] )
    gpt.final_norm.shift = assign( gpt.final_norm.shift, params["b"] )
    # weight tying: the original GPT-2 model reused the token embedding weights in the output layer to reduce the total number of parameters.
    gpt.out_head.weight  = assign( gpt.out_head.weight, params['wte'] )
    
#%%
 
if __name__ == "__main__":
    
    choice = "gpt2-small"
    cfg = GPT2CONFIG.copy()
    cfg.update( GPT2SIZE[choice] )
    cfg['qkv_bias'] = False
    cfg['drop_rate'] = 0.1
    
    model = GPTModel( cfg )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
 
    print("Token embedding layer shape:", model.tok_emb.weight.shape) # nn.Embedding(num_embeddings, embedding_dim) has weight of shape (num_embeddings, embedding_dim)
    print("Output layer shape:", model.out_head.weight.shape) # nn.Linear(in_features, out_features) has weight of shape (out_features,in_features)
 
    total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
 
    total_size_bytes = total_params * 4 # total size in bytes (assuming float32, 4 bytes per parameter)
    print(f"Total size of the model: {total_size_bytes / (1024 * 1024):.2f} MB")
    
    torch.manual_seed(123)
 
    x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
    mha = MultiHeadAttention(d_in=768, d_out=768, context_length=1024, dropout=0.0, n_heads=12, is_causal=True)
    print("context_vecs.shape:", mha(x).shape )
 
    block = TransformerBlock( cfg )
    print("Input shape:", x.shape)
    print("Output shape:", block(x).shape)
 
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append( torch.tensor(tokenizer.encode(txt1)) )
    batch.append( torch.tensor(tokenizer.encode(txt2)) )
    batch = torch.stack(batch, dim=0)
    print("Input batch:\n", batch)
 
    out = model( batch )    
    print("\nOutput shape:", out.shape)
 
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
 
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)
 
    model.eval() # disable dropout
    out = generate( model=model, idx=encoded_tensor, max_new_tokens=6, context_size=cfg["context_length"] )
 
    print("Output:", out)
    print( tokenizer.decode(out.squeeze(0).tolist()) )
    
else: 
    print(f'GPT imported from local file "{__name__}.py"')
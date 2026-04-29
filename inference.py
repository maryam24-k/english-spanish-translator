#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIS433 Project 3 — Inference Script
English → Spanish Transformer Translator

Usage:
    python inference.py

The script will prompt you to enter an English sentence and print the Spanish translation.
All model files must be in the same directory as this script (or adjust MODEL_DIR below).
"""

import os
import sys
import pickle
import torch

# ── Path setup: point to the rui package ────────────────────────────────────
# Adjust this path to wherever rui/ lives on the grading machine.
# If rui/ is in the same folder as inference.py, this works as-is.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from rui.torch.transformer import Transformer

# ── Model files (must be in same folder as this script) ─────────────────────
MODEL_DIR         = SCRIPT_DIR
WEIGHTS_FILE      = os.path.join(MODEL_DIR, 'translator_weights.pt')
VECTORIZERS_FILE  = os.path.join(MODEL_DIR, 'translator_vectorizers.pkl')

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model_and_vectorizers():
    """Load vectorizers and model weights from disk."""
    # Load vectorizers + metadata
    with open(VECTORIZERS_FILE, 'rb') as f:
        vdata = pickle.load(f)

    source_vectorizer = vdata['source_vectorizer']
    spa_index_lookup  = vdata['spa_index_lookup']
    start_idx         = vdata['start_idx']
    end_idx           = vdata['end_idx']
    seq_len           = vdata['seq_len']
    vocab_size        = vdata['vocab_size']

    # Rebuild model with same architecture used during training
    model = Transformer(
        n_layers       = 2,
        d_emb          = 128,
        n_heads        = 8,
        d_ff           = 512,
        src_vocab_size = vocab_size,
        tgt_vocab_size = vocab_size,
        seq_len        = seq_len + 1,
        dropout        = 0.1,
        prenorm        = True,
    ).to(device)

    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    model.eval()

    return model, source_vectorizer, spa_index_lookup, start_idx, end_idx, seq_len


def translate(sentence, model, source_vectorizer, spa_index_lookup,
              start_idx, end_idx, seq_len, max_len=20):
    """
    Translate an English sentence to Spanish using greedy decoding.

    Args:
        sentence (str): English input sentence.
        model: Loaded Transformer model.
        source_vectorizer: Fitted TextVectorizer for English.
        spa_index_lookup (dict): Maps token index → Spanish word.
        start_idx (int): Index of [start] token.
        end_idx (int): Index of [end] token.
        seq_len (int): Maximum sequence length.
        max_len (int): Maximum number of tokens to generate.

    Returns:
        str: Translated Spanish sentence.
    """
    with torch.no_grad():
        # Vectorize source sentence
        src = source_vectorizer(sentence)                          # numpy (1, seq_len)
        src = torch.tensor(src, dtype=torch.long).to(device)      # (1, seq_len)

        # Autoregressive decoding starting from [start]
        decoded = [start_idx]

        for _ in range(max_len):
            # Build target input: decoded tokens so far, padded to seq_len
            tgt_seq        = decoded[:seq_len]
            tgt_seq_padded = tgt_seq + [0] * (seq_len - len(tgt_seq))
            tgt = torch.tensor([tgt_seq_padded], dtype=torch.long).to(device)  # (1, seq_len)

            logits     = model((src, tgt))                              # (1, seq_len, vocab_size)
            next_token = torch.argmax(logits[0, len(decoded) - 1, :]).item()

            if next_token == end_idx or next_token == 0:
                break
            decoded.append(next_token)

        # Decode indices → words (skip the leading [start])
        words = [spa_index_lookup.get(idx, '') for idx in decoded[1:]]
        return ' '.join(w for w in words if w)


def main():
    print('Loading model...')
    model, source_vectorizer, spa_index_lookup, start_idx, end_idx, seq_len = \
        load_model_and_vectorizers()
    print('Model loaded. Ready to translate!\n')
    print('Type an English sentence and press Enter. Type "quit" to exit.\n')

    while True:
        try:
            sentence = input('English: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye!')
            break

        if not sentence:
            continue
        if sentence.lower() == 'quit':
            print('Goodbye!')
            break

        translation = translate(
            sentence, model, source_vectorizer,
            spa_index_lookup, start_idx, end_idx, seq_len
        )
        print(f'Spanish: {translation}\n')


if __name__ == '__main__':
    main()

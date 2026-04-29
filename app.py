#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIS433 Project 3 — English → Spanish Translator
Streamlit App
"""

import os
import sys
import pickle
import torch
import streamlit as st
import gdown

# ── rui path setup ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from rui.torch.transformer import Transformer

WEIGHTS_FILE     = os.path.join(SCRIPT_DIR, 'translator_weights.pt')
VECTORIZERS_FILE = os.path.join(SCRIPT_DIR, 'translator_vectorizers.pkl')

# Google Drive file ID for translator_weights.pt
WEIGHTS_GDRIVE_ID = '1ezH3a7P9odmVdFZJC33kIDRCUb4kA95B'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def download_weights_if_missing():
    """Download model weights from Google Drive if not present locally."""
    if not os.path.exists(WEIGHTS_FILE):
        st.info('Downloading model weights from Google Drive (first launch only)...')
        url = f'https://drive.google.com/uc?id={WEIGHTS_GDRIVE_ID}'
        gdown.download(url, WEIGHTS_FILE, quiet=False)
        st.success('Weights downloaded!')

# ── Load model (cached — only runs once per session) ─────────────────────────
@st.cache_resource
def load_model():
    download_weights_if_missing()

    with open(VECTORIZERS_FILE, 'rb') as f:
        vdata = pickle.load(f)

    source_vectorizer = vdata['source_vectorizer']
    spa_index_lookup  = vdata['spa_index_lookup']
    start_idx         = vdata['start_idx']
    end_idx           = vdata['end_idx']
    seq_len           = vdata['seq_len']
    vocab_size        = vdata['vocab_size']

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
              start_idx, end_idx, seq_len, max_len=50):
    with torch.no_grad():
        src = source_vectorizer(sentence)
        src = torch.tensor(src, dtype=torch.long).to(device)
        decoded = [start_idx]
        for _ in range(max_len):
            tgt_seq        = decoded[:seq_len]
            tgt_seq_padded = tgt_seq + [0] * (seq_len - len(tgt_seq))
            tgt    = torch.tensor([tgt_seq_padded], dtype=torch.long).to(device)
            logits = model((src, tgt))
            next_token = torch.argmax(logits[0, len(decoded) - 1, :]).item()
            if next_token == end_idx or next_token == 0:
                break
            decoded.append(next_token)
        words = [spa_index_lookup.get(idx, '') for idx in decoded[1:]]
        return ' '.join(w for w in words if w)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = 'EN → ES Translator',
    page_icon  = '🌐',
    layout     = 'centered',
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .title-block {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .title-block h1 { font-size: 2.5rem; margin-bottom: 0.2rem; }
    .title-block p  { color: #888; font-size: 0.95rem; }
    .translation-box {
        background: #1e1e2e;
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1.5rem;
        font-size: 1.4rem;
        font-weight: 500;
        color: #cdd6f4;
        min-height: 80px;
        margin-top: 0.5rem;
    }
    .example-header { color: #888; font-size: 0.85rem; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🌐 EN → ES Translator</h1>
    <p>Transformer trained from scratch · CIS433 Project 3 · Simon Business School</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner('Loading model...'):
    model, source_vectorizer, spa_index_lookup, start_idx, end_idx, seq_len = load_model()

# ── Main translation UI ───────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap='large')

with col1:
    st.markdown('#### 🇺🇸 English')
    user_input = st.text_area(
        label            = 'English input',
        placeholder      = 'Type an English sentence...',
        height           = 120,
        label_visibility = 'collapsed',
        key              = 'input_box',
    )
    translate_btn = st.button('Translate →', type='primary', use_container_width=True)

with col2:
    st.markdown('#### 🇪🇸 Spanish')
    if translate_btn and user_input.strip():
        result = translate(
            user_input.strip(), model, source_vectorizer,
            spa_index_lookup, start_idx, end_idx, seq_len
        )
        st.markdown(f'<div class="translation-box">{result}</div>', unsafe_allow_html=True)
    elif translate_btn and not user_input.strip():
        st.warning('Please enter a sentence.')
    else:
        st.markdown(
            '<div class="translation-box" style="color:#555;">Translation will appear here...</div>',
            unsafe_allow_html=True
        )

st.divider()

# ── Example sentences ─────────────────────────────────────────────────────────
st.markdown('<p class="example-header">✨ TRY AN EXAMPLE</p>', unsafe_allow_html=True)

examples = [
    'What time is it?',
    'Good morning!',
    'I love to learn new things.',
    'Can you help me?',
    'The weather is beautiful today.',
    'Where is the train station?',
    'I am hungry.',
    'How much does it cost?',
    'See you tomorrow.',
    'Thank you very much.',
    'I do not understand.',
    'My name is Maria.',
]

cols = st.columns(3)
for i, example in enumerate(examples):
    with cols[i % 3]:
        if st.button(example, key=f'ex_{i}', use_container_width=True):
            result = translate(
                example, model, source_vectorizer,
                spa_index_lookup, start_idx, end_idx, seq_len
            )
            st.success(f'**ES:** {result}')

st.divider()
st.caption(
    'Built with PyTorch · Trained on ~118k English-Spanish sentence pairs · '
    'Greedy decoding · Best for short phrases'
)

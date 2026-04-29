# 🌐 English → Spanish Neural Translator

A sequence-to-sequence neural machine translator built entirely from scratch using a Transformer architecture. No translation APIs, no pre-trained models — just raw PyTorch and 118,000 sentence pairs.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-username-en-es-translator.streamlit.app)

---

## 🧠 How It Works

This translator uses the classic **Encoder-Decoder Transformer** architecture introduced in *"Attention Is All You Need"* (Vaswani et al., 2017):

1. The **Encoder** reads the English sentence and produces a rich contextual representation
2. The **Decoder** generates the Spanish translation one token at a time, attending to the encoder's output at each step (cross-attention)
3. At inference time, decoding is **autoregressive** — the model feeds its own previous predictions back in until it produces an `[end]` token

### Model Configuration
| Hyperparameter | Value |
|---|---|
| Embedding dimension (`d_emb`) | 128 |
| Encoder/Decoder layers (`n_layers`) | 2 |
| Attention heads (`n_heads`) | 8 |
| Feed-forward dimension (`d_ff`) | 512 |
| Vocabulary size | 15,000 |
| Max sequence length | 50 tokens |
| Dropout | 0.1 |

### Training Details
| Setting | Value |
|---|---|
| Dataset | [ManyThings English-Spanish](http://www.manythings.org/anki/) (~118k pairs) |
| Optimizer | Adam (`β1=0.9, β2=0.98, ε=1e-9`) |
| Learning rate schedule | Linear warmup (3 epochs) + cosine decay |
| Loss | Cross-entropy with label smoothing (0.1) |
| Epochs | 30 |
| Validation accuracy | ~68% (masked token accuracy) |

---

## 🚀 Run the App Locally

### 1. Clone the repo
```bash
git clone https://github.com/your-username/en-es-translator.git
cd en-es-translator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

### 4. Or run the command-line translator
```bash
python inference.py
```

---

## 📁 Project Structure

```
en-es-translator/
├── app.py                       # Streamlit web app
├── inference.py                 # Command-line inference script
├── requirements.txt
├── translator_weights.pt        # Trained model weights
├── translator_vectorizers.pkl   # Fitted tokenizers + vocabulary
└── rui/                         # Custom transformer library
    ├── utils.py                 # TextVectorizer
    └── torch/
        ├── transformer.py       # Transformer, Encoder, Decoder, etc.
        └── utils.py             # train(), evaluate(), plotEpoch()
```

---

## 💡 Using the Command-Line Interface

```bash
$ python inference.py

Loading model...
Model loaded. Ready to translate!

English: What time is it?
Spanish: a qué hora es

English: I love to learn new things.
Spanish: me encanta aprender cosas nuevas

English: quit
Goodbye!
```

---

## ⚙️ Technical Notes

- **Tokenization**: Custom `TextVectorizer` with lowercasing and punctuation stripping. Spanish targets are wrapped with `[start]` and `[end]` tokens.
- **Padding**: Sequences shorter than `seq_len` are zero-padded; loss and accuracy ignore padding positions (masked).
- **Teacher forcing**: Used during training — the decoder receives ground-truth previous tokens as input. At inference, it receives its own predictions (greedy decoding).
- **Saving**: Both model weights (`state_dict`) and vectorizers (pickle) must be saved together for inference to work on a new machine — the vocabulary mapping is part of the pipeline.

---

## 📚 Built For

**CIS433 — Artificial Intelligence & Deep Learning**  
Simon Business School, University of Rochester  
Fulbright Scholar Project

---

## 🔗 References

- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Dataset: [ManyThings.org Anki bilingual pairs](http://www.manythings.org/anki/)

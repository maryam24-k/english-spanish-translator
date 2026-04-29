import numpy as np
from collections import Counter
import re
import pickle
 
#%%
 
class TextVectorizer:
    def __init__(self, max_tokens=20000, output_mode="int", output_sequence_length=None, ngrams=None, standardize=None):
        self.max_tokens = max_tokens
        self.output_mode = output_mode
        self.output_sequence_length = output_sequence_length
        self.ngrams = ngrams
        self.standardize = standardize
        self.vocab = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.idf_weights = None
 
    def _tokenize(self, text):
        if self.standardize:
            text = self.standardize(text)
        else:
            text = text.lower()
            text = re.sub(r'', ' ', text)
            text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        if self.ngrams and self.ngrams >= 2:
            ngram_tokens = []
            for n in range(1, self.ngrams + 1):
                for i in range( len(tokens) - n + 1 ):
                    ngram_tokens.append(' '.join(tokens[i:i + n]))
            return ngram_tokens
        return tokens
 
    def adapt(self, texts):
        """Build vocabulary from texts"""
        token_counter = Counter()
        doc_freq = Counter()  # for TF-IDF
        num_docs = 0
        print("Adapting TextVectorizer...")
        for text in texts:
            num_docs += 1
            tokens = self._tokenize(text)
            token_counter.update(tokens)
            doc_freq.update(set(tokens))
        # Reserve 0 for padding, 1 for unknown
        most_common = token_counter.most_common(self.max_tokens - 2 if self.max_tokens else None) # minus 2 to account for padding and unknown
        self.vocab = ['', ''] + [word for word, _ in most_common] # 0 for padding, 1 for unknown
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        # Compute IDF weights for tf-idf mode
        if self.output_mode == "tf_idf":
            self.idf_weights = np.zeros(len(self.vocab))
            for word, idx in self.word_to_idx.items():
                df = doc_freq.get(word, 0) + 1  # add 1 for smoothing
                self.idf_weights[idx] = np.log(num_docs / df) + 1
        print(f"Vocabulary size: {len(self.vocab)}")
 
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self.output_mode == "int":
            return self._vectorize_int(texts)
        elif self.output_mode == "multi_hot":
            return self._vectorize_multi_hot(texts)
        elif self.output_mode == "tf_idf":
            return self._vectorize_tfidf(texts)
 
    def _vectorize_int(self, texts):
        batch_indices = []
        for text in texts:
            tokens = self._tokenize(text)
            indices = [self.word_to_idx.get(t, 1) for t in tokens]  # 1 is 
            if self.output_sequence_length:
                if len(indices) < self.output_sequence_length:
                    indices = indices + [0] * (self.output_sequence_length - len(indices))
                else:
                    indices = indices[:self.output_sequence_length]
            batch_indices.append(indices)
        return np.array(batch_indices)
 
    def _vectorize_multi_hot(self, texts):
        result = np.zeros( (len(texts), len(self.vocab)), dtype=np.float32 )
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            for t in tokens:
                idx = self.word_to_idx.get(t, 1)
                if idx < len(self.vocab):
                    result[i, idx] = 1
        return result
 
    def _vectorize_tfidf(self, texts):
        result = np.zeros( (len(texts), len(self.vocab)), dtype=np.float32 )
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            for t, count in tf.items():
                idx = self.word_to_idx.get(t, 1)
                if idx < len(self.vocab):
                    result[i, idx] = count * self.idf_weights[idx]
        return result
 
    def get_vocabulary(self):
        return self.vocab
    
    def save(self, path):
        vec = {'max_tokens': self.max_tokens, 'output_sequence_length': self.output_sequence_length,
               'vocab': self.vocab, 'word_to_idx': self.word_to_idx, 'idx_to_word': self.idx_to_word,
               'has_standardizer': self.standardize is not None}
        with open(path, 'wb') as f:
            pickle.dump(vec, f)
            
    @classmethod
    def load(cls, path, standardize=None):
        data = pickle.load(path, weights_only=False)
        vectorizer = cls( max_tokens=data['max_tokens'], output_sequence_length=data['output_sequence_length'], 
                          standardize=standardize if data.get('has_standardize') else None )
        vectorizer.vocab = data['vocab']
        vectorizer.word_to_idx = data['word_to_idx']
        vectorizer.idx_to_word = data['idx_to_word']
        return vectorizer
    
if __name__ == "__main__":
    
    vectorizer = TextVectorizer( max_tokens=10, output_mode="int" )
    txts = ['how are you']
    vectorizer.adapt( txts )
    encoded = vectorizer(txts)
    print(encoded)
else:
    print(f'Importingfrom local file "{__name__}.py"')
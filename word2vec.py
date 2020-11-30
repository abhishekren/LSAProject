import numpy as np
from random import sample
# Initialization functions

data_file = "enwiki-latest-pages-articles.xml.bz2"
text_file = "en-wiki.txt"
errstring = "Must run forward inference before backpropagation"


class Embedding:

    def __init__(self, latent_dim, corpus, data_ratio=3):
        self.dim = latent_dim
        self.sigmoid = lambda x: np.divide(1, np.add(1, np.exp(-x)))
        words_seen = set()
        self.idx = dict()
        curr_idx = 0
        for word in corpus:
            if word not in words_seen:
                words_seen.add(word)
                self.idx[word] = curr_idx
                curr_idx += 1
        self.num_words = len(words_seen)
        num_data = data_ratio * self.num_words
        self.X = np.zeros(self.num_words, num_data)
        self.y = np.zeros(self.num_words, num_data)
        for i in sample(range(len(corpus)), num_data):
            start = max(0, i - 10)
            end = min(len(corpus), i + 10)
            indices = [self.idx[x] for x in corpus[start:end]]
            self.y[i, indices] = 1
            self.X[i, self.idx[corpus[i]]] = 1
        self.weights1 = np.random.rand(self.dim, self.num_words)
        self.weights2 = np.random.rand(self.num_words, self.dim)

    def fit(self, learning_rate, epochs):
        for _ in range(epochs):
            hidden = self.weights1 @ self.X
            output = self.sigmoid(self.weights2 @ hidden)
            delta = (self.y - output) * output * np.add(1, -output)
            self.weights1 -= learning_rate * hidden @ delta.T
            delta = self.weights2.T @ delta
            self.weights2 -= learning_rate * self.X @ delta.T

    def _get_one_hot(self, word):
        one_hot = np.zeros(self.num_words)
        one_hot[self.idx[word]] = 1
        return one_hot

    def word_to_center_vec(self, word):
        return self.weights1 @ self._get_one_hot(word)

    def word_to_context_vec(self, word):
        return self.weights2.T @ self._get_one_hot(word)

    def cos_similarity(self, word1, word2):
        embedding1 = self.word_to_center_vec(word1)
        embedding2 = self.word_to_context_vec(word2)
        norm1 = embedding1.T @ embedding1
        norm2 = embedding2.T @ embedding2
        return embedding1.T @ embedding2 / np.sqrt(norm1 * norm2)





# Generates a random matrix with embedding_size rows and vocabulary_size columns
# Initialize with Gaussians of mean 0 and variance 1
def initialize_word_embedding(vocabulary_size, embedding_size):
    embedding_matrix = np.random.randn(vocabulary_size, embedding_size)
    return embedding_matrix

def initalize_dense_layer(in_size, out_size):
    dense_matrix = np.random.randn(out_size, in_size)
    return dense_matrix

def initialize_everything(vocabulary_size, embedding_size):
    params={}
    params['embedding']=initialize_word_embedding(vocabulary_size, embedding_size)
    params['dense']=initalize_dense_layer(embedding_size, vocabulary_size)
    return params

# Forward Propogation
def embedding_layer(words, params):
    embedding_matrix = params['embedding']
    word_vectors = embedding_matrix[words.flatten(), :]
    return word_vectors

def dense_layer(word_vectors, params):
    dense_matrix = params['dense']
    pre_softmaxvecs = np.dot(dense_matrix, word_vectors)
    post_softmax = np.divide(np.exp(pre_softmaxvecs), np.sum(np.exp(pre_softmaxvecs)))
    return post_softmax

# Back Propogation
def calculate_error(post_softmax, labeled_data):
    dimension = post_softmax.shape[1]
    error = (-1/dimension)*np.sum(np.sum(labeled_data*post_softmax))
    return error


import numpy as np
from random import sample
import sys
import re
import nltk
from nltk.corpus import stopwords
import tqdm
from tqdm import tqdm
from gensim.corpora import WikiCorpus

# Initialization functions

data_file = r"enwiki-latest-pages-articles.xml.bz2"
text_file = r"wiki_en.txt"
errstring = "Must run forward inference before backpropagation"


class Embedding:

    def __init__(self, latent_dim, corpus, data_ratio=3):
        self.dim = latent_dim
        self.sigmoid = lambda x: np.divide(1, np.add(1, np.exp(-x)))
        self.words = set()
        self.idx = dict()
        curr_idx = 0
        for word in corpus:
            if word not in self.words:
                self.words.add(word)
                self.idx[word] = curr_idx
                curr_idx += 1
        print(f"found words {len(self.words)} words in a corpus of length {len(corpus)}")
        self.num_data = data_ratio * len(self.words)
        self.X = np.zeros((len(self.words), self.num_data))
        self.y = np.zeros((len(self.words), self.num_data))
        count = 0
        for i in sample(range(len(corpus)), self.num_data):
            start = max(0, i - 10)
            end = min(len(corpus), i + 10)
            indices = [self.idx[x] for x in corpus[start:end]]
            self.y[indices, count] = 1
            self.X[self.idx[corpus[i]], count] = 1
            count += 1
        self.weights1 = np.random.rand(self.dim, len(self.words))
        self.weights2 = np.random.rand(len(self.words), self.dim)

    def fit(self, learning_rate, epochs):
        for i in range(epochs):
            data_bar = tqdm(range(self.num_data), position=0, leave=True)
            data_bar.set_description(f"Processing epoch {i+1} out of {epochs}")
            for j in data_bar:
                hidden = self.weights1 @ self.X[:, j]
                output = self.sigmoid(self.weights2 @ hidden)
                delta = (self.y[:, j] - output) * output * np.add(1, -output)
                self.weights1 -= learning_rate * np.outer(hidden, delta)
                delta = self.weights2.T @ delta
                self.weights2 -= learning_rate * np.outer(self.X[:, j], delta)
        return self

    def _get_one_hot(self, word):
        one_hot = np.zeros(len(self.words))
        one_hot[self.idx[word]] = 1
        return one_hot

    def word_to_center_vec(self, word):
        return self.weights1 @ self._get_one_hot(word)

    def word_to_context_vec(self, word):
        return self.weights2.T @ self._get_one_hot(word)

    def cos_similarity(self, word1, word2):
        embedding1 = self.word_to_center_vec(word1)
        embedding2 = self.word_to_center_vec(word2)
        norm1 = embedding1.T @ embedding1
        norm2 = embedding2.T @ embedding2
        return embedding1.T @ embedding2 / np.sqrt(norm1 * norm2)

    def most_similar(self, word):
        others = [(other, self.cos_similarity(word, other)) for other in self.words]
        others.sort(key=lambda x: -x[1])
        return others

    def save(self):
        np.save("weights1", self.weights1)
        np.save("weights2", self.weights2)

    def load(self):
        self.weights1 = np.load("weights1")
        self.weights2 = np.load("weights1")


if __name__ == '__main__':
    file = open(text_file, 'r')
    corpus = file.read(2000000)
    corpus = re.sub(r'/\'', ' ', corpus)
    corpus = re.sub(r'[^\w\s]', '', corpus)
    corpus = re.split(r'\s', corpus)
    corpus = list(filter(''.__ne__, corpus))
    #nltk.download('stopwords')
    stops = set(stopwords.words('english'))
    corpus = [word for word in corpus if word not in stops]
    file.close()
    embed = Embedding(300, corpus, data_ratio=5).fit(0.001, 3)
    embed.save()
    print(f"Context words for 'philosopher': {embed.most_similar('philosopher')[:5]}")




# Generates a random matrix with embedding_size rows and vocabulary_size columns
# Initialize with Gaussians of mean 0 and variance 1
def initialize_word_embedding(vocabulary_size, embedding_size):
    embedding_matrix = np.random.randn(vocabulary_size, embedding_size)
    return embedding_matrix


def initalize_dense_layer(in_size, out_size):
    dense_matrix = np.random.randn(out_size, in_size)
    return dense_matrix


def initialize_everything(vocabulary_size, embedding_size):
    params = {}
    params['embedding'] = initialize_word_embedding(vocabulary_size, embedding_size)
    params['dense'] = initalize_dense_layer(embedding_size, vocabulary_size)
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
    error = (-1 / dimension) * np.sum(np.sum(labeled_data * post_softmax))
    return error

import numpy as np
from random import sample, choices
import sys
import re
import nltk
from nltk.corpus import stopwords
import tqdm
from tqdm import tqdm
import pickle
from collections import Counter
from gensim.corpora import WikiCorpus

# Initialization functions

data_file = r"enwiki-latest-pages-articles.xml.bz2"
text_file = r"wiki_en.txt"
errstring = "Must run forward inference before backpropagation"


class Embedding:
    """A class representing an autoencoder for LSA Embedding

    :param latent_dim: The dimension of the latent space in which to embed words.
    :param corpus: A list containing all the words we wish to embed.
    :param window: The length of the context word window.
    :param load_weights: Whether or not to load a pretrained model we have saved
                            from a previous training session. If a pretrained
                            model is loaded then the fit function is disabled
                            since our model will already be fit to the data.
    """

    def __init__(self, latent_dim, corpus, window=3, load_weights=False):

        # Load weights if necessary
        if load_weights:
            self.load()
            self.has_data = False
            return
        self.has_data = True

        # Define activation
        self.sigmoid = lambda x: np.divide(1, np.add(1, np.exp(-x)))

        # Separate unique words and construct one-hot encoding
        # self.idx will store the one-hot index of each word
        self.words = set(corpus)
        self.idx = {word: i for i, word in enumerate(self.words)}
        print(f"found words {len(self.words)} words in a corpus of length {len(corpus)}")

        # Initialize data to all negative examples
        self.X = np.eye(len(self.words))
        self.y = np.zeros(2 * (len(self.words),))

        # Add in all positive examples
        for i in range(len(corpus)):
            # Get the one-hot encoding indices of all words in window
            # Remove one-hot encoding index of center word
            start = max(0, i - window)
            end = min(len(corpus), i + window)
            indices = [self.idx[x] for x in corpus[start:end]]
            curr = indices.pop(i - start)
            # Add positive example (w', w) for w the center and all w' in window
            # REMEMBER: first index of y corresponds to w', second index to w
            self.y[indices, curr] += 1

        # Normalize target so that y[w, w'] is the
        # probability that w' appears within window
        # of a randomly chosen appearance of w
        self.y = self.y / np.sum(self.y, axis=0)

        # Randomly initialize embeddings
        self.weights1 = np.random.rand(latent_dim, len(self.words))
        self.weights2 = np.random.rand(len(self.words), latent_dim)

    def fit(self, lr, epochs):
        if not self.has_data: #Don't train if this is a pre-trained model!
            return
        for i in range(epochs):
            data_bar = tqdm(range(len(self.words)), position=0, leave=True)
            data_bar.set_description(f"Processing epoch {i+1} out of {epochs}")
            for j in data_bar:
                hidden = self.weights1 @ self.X[:, j]
                output = self.sigmoid(self.weights2 @ hidden)
                delta = (self.y[:, j] - output) * output * np.add(1, -output)
                self.weights1 -= lr * np.outer(hidden, delta)
                delta = self.weights2.T @ delta
                self.weights2 -= lr * np.outer(self.X[:, j], delta)
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
        kwargs = {'protocol': pickle.HIGHEST_PROTOCOL}
        np.save("weights1", self.weights1)
        np.save("weights2", self.weights2)
        with open('word_indices.pickle', 'wb') as handle:
            pickle.dump(self.idx, handle, **kwargs)

    def load(self):
        self.weights1 = np.load("weights1")
        self.weights2 = np.load("weights1")
        with open('word_indices.pickle', 'rb') as handle:
            self.idx = pickle.load(handle)


class AltEmbedding:

    def __init__(self, latent_dim, corpus, window=10):
        self.words = set(corpus)
        idx = {word: i for i, word in enumerate(self.words)}
        self.freqs = Counter([idx[word] for word in corpus])
        self.idx = {i: idx[word] for i, word in enumerate(corpus)}
        print(f"found words {len(self.words)} words in a corpus of length {len(corpus)}")
        self.center = np.random.randn(len(self.words), latent_dim)
        self.context = np.random.randn(latent_dim, len(self.words))
        self.window = window

    def fit(self, lr, epochs):
        for i in range(epochs):
            data_bar = tqdm(range(len(self.words)), position=0, leave=True)
            data_bar.set_description(f"Processing epoch {i+1} out of {epochs}")
            for j in data_bar:
                center_index = self.idx[j]
                start = max(j - self.window, 0)
                end = min(j + self.window, len(corpus) - 1) + 1
                for k in range(start, end):
                    if k == 0: next
                    context_index = self.idx[k]
                    e = np.exp(-(self.center[[center_index], :] @ self.context[:, [context_index]])[0][0])
                    self.center[[center_index], :] += lr * e / (1 + e) * self.context[:, [context_index]].T
                    self.context[:, [context_index]] += lr * e / (1 + e) * self.center[[center_index], :].T
                neg_index = choices(range(len(self.words)), weights=self.freqs, k=2*self.window)
                for k in neg_index:
                    context_index = k
                    e = np.exp(-(self.center[[center_index], :] @ self.context[:, [context_index]])[0][0])
                    self.center[[center_index], :] += lr * -1 / (1 + e) * self.context[:, [context_index]].T
                    self.context[:, [context_index]] += lr * -1 / (1 + e) * self.center[[center_index], :].T







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
    embed = Embedding(300, corpus).fit(0.001, 3)
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

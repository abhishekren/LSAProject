import numpy as np
# Initialization functions

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

# Training
def training():

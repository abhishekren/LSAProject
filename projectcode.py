import nltk
import numpy as np
from nltk.corpus import reuters
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


# black boxed word2vec model from gensim
def download_data():
    for article in reuters.fileids():
        all_articles += reuters.raw(article)

    filename=open("test.txt", "w")
    filename.write(all_articles)

def train_word2vec():
    reuters_text = open("test.txt", "r")
    model = Word2Vec(LineSentence(reuters_text), sg=0, size=100, window=5, workers=2)
    model.save("test.word2vec")

# word similarity
def word_sim(word):
    model = gensim.models.Word2Vec.load('test.word2vec')

    if word in model.wv.index2word:
        for sim_word in model.most_similar(positive=[word]):
            print(sim_word)
    else:
        print("this word is not in the corpus!")

word_sim("economy")

def 
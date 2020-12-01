import nltk
import numpy as np
from nltk.corpus import gutenberg
from nltk.corpus import reuters
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import LsiModel
from gensim.corpora import WikiCorpus
from gensim import corpora
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


# black boxed word2vec model from gensim
def download_data_as_text():
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

# word_sim("economy")

def download_as_tokens():
    all_articles=[]
    for article in reuters.fileids():
        all_articles.append(reuters.raw(article))
    tokens = []
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = set(stopwords.words('english'))
    for article in all_articles:
        temp_token = tokenizer.tokenize(article.lower())
        stopped_tokens = [i for i in temp_token if not i in en_stop]
        tokens.append(stopped_tokens)
    return tokens
    #filename=open("test2.txt", "w")
    #filename.write(all_articles)
print("Got here")



# Credit to https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python for
# helping figure out how to set up the document matrix
def train_LSIModel(tokens):
    # reuters_text = open("test2.txt", "r")
    dct = corpora.Dictionary(tokens)
    document_matrix = [dct.doc2bow(article) for article in tokens]
    print("Got here")
    model = LsiModel(document_matrix, num_topics=100, id2word=dct)
    print("Got here")
    model.save("test2.LSIModel")

# outputs example topics
def doc_sim():
    model = gensim.models.LsiModel.load('test2.LSIModel')
    print(model.print_topics(num_topics=20, num_words=10))

tokens = download_as_tokens()
train_LSIModel(tokens)
doc_sim()
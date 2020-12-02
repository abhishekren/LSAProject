import nltk
import numpy as np
from nltk.corpus import gutenberg
from nltk.corpus import reuters
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import LsiModel
from gensim.corpora import WikiCorpus
from gensim import corpora
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from multiprocessing import Process, freeze_support

print("program started...")
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
    for article in gutenberg.fileids():
        all_articles.append(gutenberg.raw(article))
    tokens = []
    tokenizer = RegexpTokenizer(r'\w+')
    stopping_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    for article in all_articles:
        article = ''.join(word for word in article if not word.isdigit())
        temp_token = tokenizer.tokenize(article.lower())
        stopped_tokens = [the_token for the_token in temp_token if not the_token in stopping_words]
        stemmed_tokens = [stemmer.stem(token) for token in stopped_tokens]
        tokens.append(stemmed_tokens)
    print("finished creating tokens")
    return tokens
    #filename=open("test2.txt", "w")
    #filename.write(all_articles)



# Credit to https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python for
# helping figure out how to set up the document matrix
def train_LSIModel(tokens, num_top):
    # reuters_text = open("test2.txt", "r")
    dct = corpora.Dictionary(tokens)
    document_matrix = [dct.doc2bow(article) for article in tokens]
    model = LsiModel(document_matrix, num_topics=num_top, id2word=dct)
    model.save("test2.LSIModel")
    return model

# outputs example topics
def doc_sim():
    model = gensim.models.LsiModel.load('test2.LSIModel')
    print(model.print_topics(num_topics=20, num_words=10))

# find coherence scores for choices of num_topics
def find_best_coherence(tokens, range_num_topics):
    dct = corpora.Dictionary(tokens)
    coherences=[]
    for i in range(range_num_topics):
        mod = train_LSIModel(tokens, i+1)
        coherences[i] = CoherenceModel(model=mod, texts=tokens, dictionary=dct, coherence='c_v').get_coherence()
        print("gets here "+str(i))
    # Find maximum
    maximum = coherences[0]
    max_index = 0
    for i in range(len(coherences)):
        if coherences[i]>maximum:
            max_index = i+1
            maximum = coherences[i]
    print(str(max_index)+" has coherence " + maximum)
    return max_index

def compute_coherence(tokens, num_topics):
    dct = corpora.Dictionary(tokens)
    if __name__ == '__main__':
        freeze_support()
        mod = train_LSIModel(tokens, num_topics)
        coh_mod = CoherenceModel(model=mod, texts=tokens, dictionary=dct, coherence='c_v')
        return coh_mod.get_coherence()
tokens = download_as_tokens()
train_LSIModel(tokens, 20)
print(compute_coherence(tokens, 20))
doc_sim()
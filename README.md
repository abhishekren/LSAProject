# Learning Objectives

  - We want students to understand the theoretical underpinnings behind models like Word2Vec and Latent Semantic Analysis
  - Students should understand how libraries such as nltk and gensim operate on a cursory level
  - Most importantly, get an introduction to the fundamentals of Natural Langauge Processing

# Navigating the repository
The coding section is located in the .ipynb document in the main folder. A student should look to this section in order to practice coding with the notebook and getting an idea of LSA and Word2Vec. The documentation is in the documentation folder and contains some notes on LSA as well as slides. A teacher could hypothetically use the slide deck in order to teach LSA and Word2Vec, whereas the notes expound on the main theoretical underpinnings behind LSA and Word2Vec. The quiz questions are located in the quiz folder and consists of 10 questions along with corresponding answers in order to spot check whether a student indeed understands the topics.
The coding section contains two parts. In the first part, students will implement an autoencoder which maps words from running text into a low-dimensional latent space. They will then try out the autoencoder on a corpus of natural langauge scraped from Wikipedia. In the second part, we show that this autoencoder is identical to a gradient-ascent scheme based on "center" and "context" embeddings. Students then implement this gradient-ascent scheme and try out the results on the Wikipedia text corpus.

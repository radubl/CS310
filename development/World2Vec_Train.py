import os, simplejson as json, csv, nltk, datetime
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import logging
from gensim.models import word2vec

dataPath = "../tweets/";

def getSentences():
  sentences = []

  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

  i = 0;

  # iterate through each user's tweets
  for fileName in os.listdir(dataPath):

    i += 1

    with open(dataPath + fileName, 'r+') as f:
      print("Started Parsing Sentences for: " + fileName)

      joinedTweets = ""

      for line in f:

        dataPayload = json.loads(line)

        raw_sentences = tokenizer.tokenize(" ".join(dataPayload['allTokensLemmatized']).strip())

        for sen in raw_sentences:
          if sen != '.' and len(sen) != 0:
            sentences.append(sen.encode('utf-8').split())

  return sentences


def trainWorld2Vec():

  sentences = getSentences()

  num_features = 300    # Word vector dimensionality                      
  min_word_count = 40   # Minimum word count                        
  num_workers = 4       # Number of threads to run in parallel
  context = 10          # Context window size                     

  # Initialize and train the model (this will take some time)
  print "Training model..."
  model = word2vec.Word2Vec(sentences, workers=num_workers,
    size=num_features, 
    min_count = min_word_count, 
    window = context)

  # If you don't plan to train the model any further, calling 
  # init_sims will make the model much more memory-efficient.
  model.init_sims(replace=True)

  # It can be helpful to create a meaningful model name and 
  # save the model for later use. You can load it later using Word2Vec.load()
  model_name = "300features_40minwords_10context"
  model.save(model_name)

def testWorld2Vec():
  model = Word2Vec.load("300features_40minwords_10context")

if __name__ == '__main__':

  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
      level=logging.INFO)

  trainWorld2Vec()
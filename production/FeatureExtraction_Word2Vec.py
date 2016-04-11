import os, simplejson as json, nltk, datetime
from gensim.models import Word2Vec as gs_w2v
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import logging
from gensim.models import word2vec
import numpy as np
dataPath = "../tweets/";

def removeStopWords(tokens):
	stopwords = set(nltk.corpus.stopwords.words('english'))

	extraStopWords = ['n\'t','amp','http','...']

	# remove stop words and words with length less than 3
	return [w for w in tokens if w not in stopwords and w not in extraStopWords and len(w) > 2 
	and not w.startswith( '@' ) and not w.startswith( 'http' )]
	

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

  model = gs_w2v.Word2Vec(sentences, workers=num_workers,
    size=num_features, 
    min_count = min_word_count, 
    window = context)

  # finished training
  model.init_sims(replace=True)

  # It can be helpful to create a meaningful model name and 
  # save the model for later use. You can load it later using Word2Vec.load()
  model_name = "300features_40minwords_10context"
  model.save(model_name)

# Function to average all of the word vectors in a given paragraph
def makeFeatureVec(words, model, num_features):
	
	featureVec = np.zeros((num_features,),dtype="float32")

	nwords = 0.

	# list that contains the names of the words in the model's vocabulary. Convert it to a set, for speed 
	index2word_set = set(model.index2word)

	# Loop over each word in the review and, if it is in the model's
	# vocaublary, add its feature vector to the total
	for word in words:
		if word in index2word_set: 
			nwords += 1.
			featureVec = np.add(featureVec,model[word])

	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec,nwords)
	return featureVec


# Given a set of tweets (each one a list of words), calculate 
# the average feature vector for each one and return a 2D numpy array 
def getAvgFeatureVecs(tweets, model, num_features):

	# Initialize a counter
	counter = 0.

	# Preallocate a 2D numpy array, for speed
	reviewFeatureVecs = np.zeros((len(tweets),num_features),dtype="float32")

	print 'Computing feature vecss'

	# Loop through the tweets
	for tweet in tweets:

	   # Call the function (defined above) that makes average feature vectors
	   reviewFeatureVecs[counter] = makeFeatureVec(tweet, model, num_features)

	   # Increment the counter
	   counter += 1.

	return reviewFeatureVecs

def Word2Vec(condition_one, condition_two):
	
	try:
		model = gs_w2v.load("300features_40minwords_10context")
	except Exception, e:
		trainWorld2Vec()
	finally:
		model = gs_w2v.load("300features_40minwords_10context")


	num_features = 300

	unigrams = []
	conditions = []
	i = 0;

	# iterate through each user's tweets
	for fileName in os.listdir(dataPath):

		i += 1

		with open(dataPath + fileName, 'r+') as f:
			print("Started Parsing: " + fileName)

			joinedTweets = ""

			for line in f:

				dataPayload = json.loads(line)

				condition = dataPayload['metadata']['condition']

				if condition == condition_one:
					conditions.append(1)
				elif condition == condition_two:
					conditions.append(0)
				else:
					continue

				userCorpus = dataPayload['allTokensLemmatized']	

				# removing stop words from the entire corupus
				userCorpus = removeStopWords(userCorpus)

				unigrams.append(" ".join(userCorpus))

	featureSet = getAvgFeatureVecs( unigrams, model, num_features )

	return (featureSet, conditions)


if __name__ == '__main__':

	print Word2Vec("control", "depression")
from __future__ import division
import os, pickle, nltk, datetime
from gensim.models import Word2Vec
import logging
import numpy as np
import Create_W2V_model

dataPath = "../stopwords_removed_tweets_pickles/"
metadataPath = '../user-conditions.p'

def removeStopWords(tokens):
	stopwords = set(nltk.corpus.stopwords.words('english'))

	extraStopWords = ['n\'t','amp','http','...']

	# remove stop words and words with length less than 3
	return [w for w in tokens if w not in stopwords and w not in extraStopWords and len(w) > 2 
	and not w.startswith( '@' ) and not w.startswith( 'http' )]


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

	print 'Computing feature vectors'

	# Loop through the tweets
	for tweet in tweets:

		 # Call the function (defined above) that makes average feature vectors
		 reviewFeatureVecs[counter] = makeFeatureVec(tweet, model, num_features)

		 # Increment the counter
		 counter += 1.

	return reviewFeatureVecs

def runWord2Vec(condition_one, condition_two):
	num_features = 300
	
	try:
		model = Word2Vec.load("../models/W2V/"+ str(num_features) + "features_20minwords_10context")
	except Exception, e:
		Create_W2V_model.trainWorld2Vec(num_features)
		model = Word2Vec.load("../models/W2V/"+ str(num_features) + "features_20minwords_10context")

	print model.most_similar("good")
	print model.most_similar("queen")
	print model.most_similar("google")
	print model.most_similar("happy")
	print model.most_similar("drug")

	unigrams = []
	conditions = []
	i = 0;


	sentences = []
	userList = pickle.load(open(metadataPath, "rb" ))

	i = 0;

	for userData in userList:
		i += 1
		name = userData[0]
		condition = userData[1]

		if condition == condition_one:
			conditions.append(1)
		elif condition == condition_two:
			conditions.append(0)
		else:
			continue

		print str(i) + ". Opening " + dataPath + name + '.p, \n Progress: ' + str(int(i/len(userList) * 100)) + '%'


		tweetsByWeek = pickle.load(open(dataPath + name + '.p', "rb" ))


		for label in tweetsByWeek:
			for sentence in tweetsByWeek[label]:
				words = sentence.split(" ")
				unigrams.append(words)

	featureSet = getAvgFeatureVecs( unigrams, model, num_features )

	data = (featureSet, conditions)

	fileName = condition_one + '_' + condition_two + '_' + str(num_features)

	pickle.dump(data, open( "../feature_vectors/word_2_vec/" + fileName +".p", "wb" ) )
	print "Dumping pickle to " + fileName + ".p"

if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
		level=logging.INFO)

	runWord2Vec("control", "depression")
	runWord2Vec("ptsd", "depression")
	runWord2Vec("control", "ptsd")
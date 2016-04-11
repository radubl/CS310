from __future__ import division
import os, pickle, nltk, datetime
from gensim.models import Word2Vec
import logging
import numpy as np
import Create_W2V_model
from sklearn.cluster import KMeans

dataPath = "../stopwords_removed_tweets_pickles/"
metadataPath = '../user-conditions.p'

def removeStopWords(tokens):
	stopwords = set(nltk.corpus.stopwords.words('english'))

	extraStopWords = ['n\'t','amp','http','...']

	# remove stop words and words with length less than 3
	return [w for w in tokens if w not in stopwords and w not in extraStopWords and len(w) > 2 
	and not w.startswith( '@' ) and not w.startswith( 'http' )]

def create_bag_of_centroids( wordlist, word_centroid_map ):

	num_centroids = max( word_centroid_map.values() ) + 1
	
	# Pre-allocate the bag of centroids vector (for speed)
	bag_of_centroids = np.zeros( num_centroids, dtype="float32" )

	for word in wordlist:
		if word in word_centroid_map:
			index = word_centroid_map[word]
			bag_of_centroids[index] += 1

	return bag_of_centroids


def runWord2Vec(condition_one, condition_two,word_centroid_map):

	i = 0;

	unigrams = []
	conditions = []

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
				unigrams += words

		featureSet.append(create_bag_of_centroids( unigrams, word_centroid_map))

	data = (featureSet, conditions)

	fileName = 'bog_' + condition_one + '_' + condition_two + '_' + str(num_features)

	pickle.dump(data, open( "../feature_vectors/word_2_vec/" + fileName +".p", "wb" ) )
	print "Dumping pickle to " + fileName + ".p"

if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
		level=logging.INFO)
	num_features = 300
	
	try:
		model = Word2Vec.load("../models/W2V/"+ str(num_features) + "features_20minwords_10context")
	except Exception, e:
		Create_W2V_model.trainWorld2Vec(num_features)
		model = Word2Vec.load("../models/W2V/"+ str(num_features) + "features_20minwords_10context")

	# average of 100 words per cluster
	word_vectors = model.syn0
	num_clusters = int(word_vectors.shape[0] / 100)

	# Initalize a k-means object and use it to extract centroids
	print "Running KMeans Clustering"
	spec_clustering = KMeans( n_clusters = num_clusters, n_init=3)
	idx = spec_clustering.fit_predict( word_vectors )

	word_centroid_map = dict(zip( model.index2word, idx ))

	featureSet = []
	userList = pickle.load(open(metadataPath, "rb" ))
	runWord2Vec("control", "depression",word_centroid_map)
	runWord2Vec("ptsd", "depression",word_centroid_map)
	runWord2Vec("control", "ptsd",word_centroid_map)
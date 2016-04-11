from __future__ import division
import os, pickle, nltk, datetime
from gensim.models import Word2Vec
import logging
import numpy as np

dataPath = "../tweets_by_week_pickles/"
metadataPath = '../user-conditions.p'

def getSentences():
	sentences = []
	userList = pickle.load(open(metadataPath, "rb" ))
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	sentences = []

	i = 0;

	for userData in userList:
		i += 1
		name = userData[0]
		condition = userData[1]

		print str(i) + ". Opening " + dataPath + name + '.p, \n Progress: ' + str(int(i/len(userList) * 100)) + '%'


		tweetsByWeek = pickle.load(open(dataPath + name + '.p', "rb" ))


		for label in tweetsByWeek:
			for sentence in tweetsByWeek[label]:
				sentences.append(sentence.split(" "))

	return sentences


def trainWorld2Vec(num_features):

	sentences = getSentences()
				 
	min_word_count = 20   # Minimum word count                        
	num_workers = 4       # Number of threads to run in parallel
	context = 10          # Context window size                     
	downsampling = 1e-3   # Downsample setting for frequent words

	print "Training Word2Vec model..."
	model = Word2Vec(sentences, workers=num_workers, sg=1, size=num_features, min_count = min_word_count, window = context, sample = downsampling, seed=1)

	# finished training
	model.init_sims(replace=True)

	model_name = "../models/W2V/"+str(num_features)+"features_20minwords_10context"
	model.save(model_name)


if __name__ == '__main__':

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
		level=logging.INFO)

	trainWorld2Vec(300)
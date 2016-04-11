from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

import logging, pickle, csv
import numpy as np

from gensim.models import Doc2Vec

dataPath = "../tweets_by_week_pickles/"

# Function to average all of the label vectors for a user
def makeFeatureVec(name,words, model, num_features):
	
	featureVec = np.zeros((num_features,),dtype="float32")

	nwords = 0.

	totalSize = 0.

	for label in tweetsByWeek:

		size = len(tweetsByWeek[label])

		totalSize += size

		try:
			featureVec = np.add(featureVec,np.multiply(model[name+ '_' +label], size))
		except Exception, e:
			pass

	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec,totalSize)
	return featureVec

if __name__ == '__main__':
	
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = Doc2Vec.load("../models/D2V/test_perm_10_epoch.d2v")

	userList = pickle.load(open('../user-conditions.p', "rb" ))

	train_arrays = []
	train_labels = []

	for userData in userList:
		name = userData[0]
		condition = userData[1]

		if condition == 'control':
			binaryCondition = 0;
		elif condition == 'depression':
			binaryCondition = 1;
		else:
			continue;

		nwords = 0.

		totalSize = 0.

		tweetsByWeek = pickle.load(open(dataPath + name + '.p', "rb" ))

		for label in tweetsByWeek:

			size = len(tweetsByWeek[label])

			totalSize += size

			try:
				train_arrays.append(model[name+ '_' +label])
				train_labels.append(binaryCondition)
			except Exception, e:
				pass

	# train_arrays = np.divide(train_arrays,totalSize)

	print len(train_labels)

	model = RandomForestClassifier(max_depth=5, n_estimators=100)
	scores = cross_val_score(model, train_arrays, train_labels, cv=10)

	print scores.mean(), scores.std()


# condition/label: 0.672161641891 6.77779333329e-05 testA_10_epochs
# 					0.672645229275 6.54270280606e-05
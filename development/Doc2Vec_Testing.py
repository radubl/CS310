from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

import logging, pickle, csv
import numpy as np

from gensim.models import Doc2Vec

dataPath = "../tweets_by_week_pickles/"

if __name__ == '__main__':
	
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = Doc2Vec.load("../models/test_perm_10_epoch.d2v")

	with open('../user-metadata.csv', 'rb') as f:
	    reader = csv.reader(f)
	    userList = list(reader)

	userList.pop(0)

	train_arrays = []
	train_labels = []

	for userData in userList:
		name = userData[0]
		condition = userData[4]

		if condition == 'control':
			binaryCondition = 0;
		elif condition == 'depression':
			binaryCondition = 1;
		else:
			continue;

		try:
			tweetsByWeek = pickle.load(open(dataPath + name + '.p', "rb" ))

			for label in tweetsByWeek:
				train_arrays.append(model[name+ '_' +label])
				train_labels.append(binaryCondition)
		except Exception, e:
			pass

	model = RandomForestClassifier(max_depth=5, n_estimators=100)
	scores = cross_val_score(model, train_arrays, train_labels, cv=10)

	print scores.mean(), scores.std()







# for i in range(12500):
#     prefix_train_pos = 'TRAIN_POS_' + str(i)
#     prefix_train_neg = 'TRAIN_NEG_' + str(i)
#     train_arrays[i] = model[prefix_train_pos]
#     train_arrays[12500 + i] = model[prefix_train_neg]
#     train_labels[i] = 1
#     train_labels[12500 + i] = 0
# 0.672161641891 6.77779333329e-05 testA_10_epochs
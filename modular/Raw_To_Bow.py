from __future__ import division
import os, pickle, nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

dataPath = "../tweets_by_week_pickles/"
metadataPath = '../user-conditions.p'


def BagOfWords(condition_one, condition_two, maxFeatures):


	userList = pickle.load(open(metadataPath, "rb" ))

	BagOfWords = []
	conditions = []

	lemmatizer = WordNetLemmatizer()

	i = 0;

	for userData in userList:
		i += 1
		name = userData[0]
		condition = userData[1]

		if condition == condition_one:
			binaryCondition = 0;
		elif condition == condition_two:
			binaryCondition = 1;
		else:
			continue;


		print str(i) + ". Opening " + dataPath + name + '.p, \n Progress: ' + str(i/len(userList) * 100) + '%'


		tweetsByWeek = pickle.load(open(dataPath + name + '.p', "rb" ))

		wordsByUser = ""

		for label in tweetsByWeek:
			for sentence in tweetsByWeek[label]:
				wordsByUser += lemmatizer.lemmatize(sentence.decode('utf-8')) + " "

		BagOfWords.append(wordsByUser)
		conditions.append(binaryCondition)


	vectorizer = CountVectorizer(analyzer = "word",   \
				 tokenizer = None,    \
				 preprocessor = None, \
				 max_features = maxFeatures,
				 max_df=0.95, 					# words appearing in more than 95% of the documents are removed
				 min_df=2, 						# words appearing in only one document are removed
                 stop_words='english')

	featureSet = vectorizer.fit_transform(BagOfWords)

	featureSet = featureSet.toarray()

	data = (featureSet, conditions)

	fileName = condition_one + '_' + condition_two + '_' + str(maxFeatures)

	pickle.dump(data, open( "../feature_vectors/bag_of_words/" + fileName +".p", "wb" ) )
	print "Dumping pickle to " + fileName + ".p"


if __name__ == '__main__':

	BagOfWords("control", "depression",10000)
	BagOfWords("ptsd", "depression",10000)
	BagOfWords("control", "ptsd",10000)
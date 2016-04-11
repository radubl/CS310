import os, simplejson as json, csv, nltk, datetime
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
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

def testWorld2Vec(condition_one, condition_two, output_path):
	model = Word2Vec.load("300features_40minwords_10context")

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

	dataVecs = getAvgFeatureVecs( unigrams, model, num_features )

	forest = RandomForestClassifier(n_estimators = 100)

	scores = cross_val_score(forest, dataVecs, conditions, cv=10)

	output = { datetime.datetime.now().strftime("%H:%M:%S on %B %d, %Y") : {"scores" : ", ".join(str(item) for item in scores), "mean" : str(scores.mean()), "std": str(scores.std())}}

	outputFileName = 'W2Vec_RandomForest-' + condition_one + '-' + condition_two + '.json'

	outputFileNamePath = output_path + outputFileName

	if outputFileName in os.listdir(output_path):

		data = None

		with open(outputFileNamePath) as f:
			data = json.load(f)

		data.append(output)

		with open(outputFileNamePath, 'w') as f:
			json.dump(data, f)

		print 'appending results to ' + outputFileNamePath
	else:
		with open(outputFileNamePath, 'w') as f:
	   		json.dump([output], f)
		print 'printing results to ' + outputFileNamePath

if __name__ == '__main__':

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
		level=logging.INFO)

	testWorld2Vec("control", "depression","../results_dev/")
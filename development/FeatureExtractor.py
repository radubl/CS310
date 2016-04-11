import os, simplejson as json, csv, nltk
from nltk.collocations import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score as cvs


dataPath = "../tweets/";

def removeStopWords(tokens):
	stopwords = set(nltk.corpus.stopwords.words('english'))

	extraStopWords = ['n\'t','amp','http','...']

	# remove stop words and words with length less than 3
	return [w for w in tokens if w not in stopwords and w not in extraStopWords and len(w) > 2 
	and not w.startswith( '@' ) and not w.startswith( 'http' )]

def findBigrams():
	tokenizer = Tokenizer(preserve_case=False)

	for fileName in os.listdir(dataPath):

		with open(dataPath + fileName, 'r+') as f:
			print("Started Parsing: " + fileName)

			joinedTweets = ""

			for line in f:
				dataPayload = json.loads(line);
				userCorpus = dataPayload['allTokensLemmatized']
				
				bigram_measures = nltk.collocations.BigramAssocMeasures()

				finder = BigramCollocationFinder.from_words(userCorpus)

				# print 10 most common bigrams
				print finder.nbest(bigram_measures.pmi, 10)

		return

def findUnigrams():

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

				if condition == "ptsd":
					print "PTSD"
					continue

				userCorpus = dataPayload['allTokensLemmatized']

				# removing stop words from the entire corupus
				userCorpus = removeStopWords(userCorpus)

				unigrams.append(" ".join(userCorpus))

				if condition == "control":
					conditions.append(1)
				else:
					conditions.append(0)

	forest = RandomForestClassifier(n_estimators = 100)

	# forest = forest.fit(train_data_features, train["sentiment"] )

	vectorizer = CountVectorizer(analyzer = "word",   \
				 tokenizer = None,    \
				 preprocessor = None, \
				 stop_words = None,   \
				 max_features = 5000)

	train_data_features = vectorizer.fit_transform(unigrams)

	train_data_features = train_data_features.toarray()

	score = cvs(forest, train_data_features, conditions, cv=10)

	print score

	# with open('../outputs/unigrams.json', 'w') as outfile:
 #   		json.dump(featureDict, outfile)
	print 'done writing unigrams ' + str(i)


if __name__ == '__main__':

	findUnigrams()
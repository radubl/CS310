import os, simplejson as json, nltk
from sklearn.feature_extraction.text import CountVectorizer


dataPath = "../tweets/";

def removeStopWords(tokens):
	stopwords = set(nltk.corpus.stopwords.words('english'))

	extraStopWords = ['n\'t','amp','http','...']

	# remove stop words and words with length less than 3
	return [w for w in tokens if w not in stopwords and w not in extraStopWords and len(w) > 2 
	and not w.startswith( '@' ) and not w.startswith( 'http' )]


def BagOfWords(condition_one, condition_two, maxFeatures):

	print "Building Unigram frequencies between " + condition_one + " and " + condition_two + ", max_features = " + str(maxFeatures)

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

		### uncomment for debugging:
		# if i > 20:
		# 	pass
		# break

	vectorizer = CountVectorizer(analyzer = "word",   \
				 tokenizer = None,    \
				 preprocessor = None, \
				 stop_words = None,   \
				 max_features = maxFeatures)

	featureSet = vectorizer.fit_transform(unigrams)

	featureSet = featureSet.toarray()

	return (featureSet, conditions)


if __name__ == '__main__':

	print BagOfWords("control", "depression",5000)
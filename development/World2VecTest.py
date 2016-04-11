import os, simplejson as json, csv, nltk, datetime
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

dataPath = "../tweets/";

def removeStopWords(tokens):
	stopwords = set(nltk.corpus.stopwords.words('english'))

	extraStopWords = ['n\'t','amp','http','...']

	# remove stop words and words with length less than 3
	return [w for w in tokens if w not in stopwords and w not in extraStopWords and len(w) > 2 
	and not w.startswith( '@' ) and not w.startswith( 'http' )]

def World2Vec(condition_one, condition_two, output_path):

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


	forest = RandomForestClassifier(n_estimators = 100)

	vectorizer = CountVectorizer(analyzer = "word",   \
				 tokenizer = None,    \
				 preprocessor = None, \
				 stop_words = None,   \
				 max_features = 5000)

	train_data_features = vectorizer.fit_transform(unigrams)

	train_data_features = train_data_features.toarray()

	scores = cross_val_score(forest, train_data_features, conditions, cv=10)

	output = { datetime.datetime.now().strftime("%H:%M:%S on %B %d, %Y") : {"scores" : ", ".join(str(item) for item in scores), "mean" : str(scores.mean()), "std": str(scores.std())}}

	outputFileName = 'RandomForest-' + condition_one + '-' + condition_two + '.json'

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

	World2Vec("control", "depression","../results/")
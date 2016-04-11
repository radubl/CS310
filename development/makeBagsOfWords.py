import gensim, os, nltk, simplejson as json, re, pickle, datetime, string

dataPath = "../tweets/";
outputDataPath = "../tweets_by_week_pickles/"

def removeStopWords(tokens):
	stopwords = set(nltk.corpus.stopwords.words('english'))

	extraStopWords = ['n\'t','amp','http','...']

	# remove stop words and words with length less than 3
	filteredSet =  [w for w in tokens if w not in stopwords and 
										 w not in extraStopWords 
										 and len(re.sub('[^A-Za-z0-9]+', '', w)) > 3 	# the word without spaces and punctuation
										 and 'http' not in w
										 and not w.startswith( '@' )]
	return filteredSet

def removeLinksAnduserMentions(tokens):

	# remove stop words and words with length less than 3
	filteredSet =  [w for w in tokens.split(" ") if 'http' not in w and '@' not in w]
	return " ".join(filteredSet)


def makeBagOfWordsFiles(condition_one, condition_two):
	unigrams = []
	conditions = []
	i = 0;
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

				unigrams.append(" ".join(userCorpus).encode('utf-8'))


	outputFileNamePath = 'bagOfWords-' + condition_one + '-' + condition_two + '.json'

	data = {'unigrams' : unigrams, 'conditions' : conditions }

	with open(outputFileNamePath, 'w') as f:
		json.dump(data, f)
		print 'finished writing to ' + outputFileNamePath

def getTweetsByWeek():
	unigrams = []
	conditions = []

  	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	i = 0;

	for fileName in os.listdir(dataPath):

		i += 1

		with open(dataPath + fileName, 'r+') as f:
			print("Started Parsing: " + fileName)	

			for line in f:

				dataPayload = json.loads(line)

				userCorpus = dataPayload['data']
				condition = dataPayload['metadata']['condition']

				tweetsByWeeks = {}

				for timeTextPair in userCorpus:
					dateObject = datetime.datetime.strptime(timeTextPair['time'], "%a %b %d %H:%M:%S +0000 %Y")

					week_id = str(dateObject.isocalendar()[0]) + '_' + str(dateObject.isocalendar()[1])
					text = timeTextPair['text']

					text = removeLinksAnduserMentions(text)

					text = tokenizer.tokenize(text.strip())

					exclude = set(string.punctuation)
					text = ''.join(ch for ch in " ".join(text).encode('utf-8') if ch not in exclude)

					if week_id in tweetsByWeeks:
						tweetsByWeeks[week_id].append(text)
					else:
						tweetsByWeeks[week_id] = [text]

				pickle.dump( tweetsByWeeks, open( outputDataPath + fileName[:-5] + ".p", "wb" ) )
				print "Dumping pickle to " +  outputDataPath + fileName[:-5] + '.p'

def removeStopWordsFromPickles():
	path = '../stopwords_removed_tweets_pickles/'

	i = 0;

	for fileName in os.listdir(outputDataPath):

		i += 1;

		tweetsByWeek = pickle.load(open(outputDataPath + fileName, "rb" ))

		newFile = {}

		for label in tweetsByWeek:
			removedStopWords = []
			for sentence in tweetsByWeek[label]:
				words = sentence.split(' ')
				words = removeStopWords(words)
				removedStopWords.append(' '.join(words))
			newFile[label] = removedStopWords

		pickle.dump(newFile, open( path + fileName, "wb" ) )
		print str(i) + ". Dumping pickle to " +  outputDataPath + fileName


def getUserConditions():
	conditions = []

	i = 0;

	for fileName in os.listdir(dataPath):

		i += 1

		with open(dataPath + fileName, 'r+') as f:
			print("Started Parsing: " + fileName)	

			for line in f:

				dataPayload = json.loads(line)

				conditions.append((fileName[:-5],dataPayload['metadata']['condition']))

	pickle.dump( conditions, open( "../user-conditions.p", "wb" ) )
	print "Dumping pickle to ../user-conditions.p"

if __name__ == '__main__':
	# makeBagOfWordsFiles('control','depression')
	# makeBagOfWordsFiles('control','ptsd')
	# makeBagOfWordsFiles('ptsd','depression')

	# getTweetsByWeek()

	removeStopWordsFromPickles()
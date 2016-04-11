from __future__ import division
import gensim, os, nltk, pickle
import logging, bz2

dataPath = "../stopwords_removed_tweets_pickles/"
metadataPath = '../user-conditions.p'

def getTweetCorpus():
	corpus = []
	userList = pickle.load(open(metadataPath, "rb" ))

	i = 0;

	for userData in userList:
		name = userData[0]
		i += 1

		print str(i) + ". Opening " + dataPath + name + '.p, \n Progress: ' + str(int(i/len(userList) * 100)) + '%'


		tweetsByWeek = pickle.load(open(dataPath + name + '.p', "rb" ))


		for label in tweetsByWeek:
			for sentence in tweetsByWeek[label]:
				corpus.append(sentence.split(" "))

	return corpus

def createLDAModel():

	texts = getTweetCorpus()

	dictionary = gensim.corpora.Dictionary(texts)

	corpus = [dictionary.doc2bow(text) for text in texts]

	lda = gensim.models.ldamodel.LdaModel(corpus=corpus, 
		id2word=dictionary, num_topics=200, update_every=1, chunksize=10000, passes=1)

	lda.save('../models/LDA/allCorpus_200.lda')

	print lda.print_topics(2)


def makeLDAFeatures(condition_one, condition_two):

	

	# texts = [userWords.split() for userWords in unigrams]


	# # for x in xrange(1):

	# dictionary = gensim.corpora.Dictionary(texts)

	# query = 'good bad ugly'.split()

	query = dictionary.doc2bow(query)

	lda = gensim.models.ldamodel.LdaModel.load('../models/LDA/allCorpus_200.lda')

	print lda[query]

	a = list(sorted(lda[query], key=lambda x: x[1]))

	print a
	print a[0]

	print lda.print_topic(a[0][0])
	print lda.print_topic(a[-1][0])

if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	unigrams = []
	conditions = []
	# createLDAModel()
	makeLDAFeatures('control','depression')
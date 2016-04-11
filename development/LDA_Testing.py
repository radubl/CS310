import gensim, os, nltk, simplejson as json
import logging, bz2

	
def LDATesting(condition_one, condition_two):


	with open('bagOfWords-'+condition_one+'-'+condition_two+'.json', 'r+') as f:
		for line in f:
			dataPayload = json.loads(line)

			texts = dataPayload['unigrams']

			texts = [userWords.split() for userWords in texts]

			dictionary = gensim.corpora.Dictionary(texts)

			corpus = [dictionary.doc2bow(text) for text in texts]

			lda = gensim.models.ldamodel.LdaModel(corpus=corpus, 
				id2word=dictionary, num_topics=200, update_every=1, chunksize=10000, passes=1)

			lda.save('test_CvD.lda')

			print lda.print_topics(2)


def loadLda(condition_one, condition_two,filePath):
	
	with open('bagOfWords-'+condition_one+'-'+condition_two+'.json', 'r+') as f:
		for line in f:
			dataPayload = json.loads(line)

			unigrams = dataPayload['unigrams']

			texts = [userWords.split() for userWords in unigrams]

			dictionary = gensim.corpora.Dictionary(texts)

			for x in xrange(1):

				query = dictionary.doc2bow(unigrams[0].split())

				lda = gensim.models.ldamodel.LdaModel.load(filePath)

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
	LDATesting('control','depression')
	loadLda('control','depression','test_CvD.lda')
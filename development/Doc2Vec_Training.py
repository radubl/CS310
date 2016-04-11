import logging, pickle, os 
from gensim import utils
from gensim.models.doc2vec import LabeledSentence

# numpy
import numpy

# random
from random import shuffle
from gensim.models import Doc2Vec

dataPath = "../tweets_by_week_pickles/"

class LabeledLineSentence(object):
	def __init__(self):
		pass
	
	def __iter__(self):
		for fileName in os.listdir(dataPath):
			userTexts = pickle.load( open( dataPath + fileName, "rb" ) )
			for label in userTexts:
				for sentence in userTexts[label]:
					yield LabeledSentence(utils.to_unicode(sentence).split(), [fileName[:-2] + '_%s' % label])
	
	def to_array(self):
		self.sentences = []
		for fileName in os.listdir(dataPath):
			userTexts = pickle.load( open( dataPath + fileName, "rb" ) )
			for label in userTexts:
				for sentence in userTexts[label]:
					self.sentences.append(LabeledSentence(utils.to_unicode(sentence + " " + fileName[:-2] + '_%s' % label).split(), [fileName[:-2] + '_%s' % label]))
		return self.sentences
	
	def sentences_perm(self):
		shuffle(self.sentences)
		return self.sentences

def trainDoc2Vec():
	pass

if __name__ == '__main__':

	sentences = LabeledLineSentence()
	
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	try:
		print "Trying to load model"
		model = Doc2Vec.load("../models/test_perm_10_epoch.d2v")
	except Exception, e:

		print "Model not found, constructin model with size 300, window 30, alpha 0.025, 5 iterations"
		model = Doc2Vec(min_count=3, window=30, size=300, sample=1e-4, negative=5, alpha=0.025, min_alpha=0.025, workers=4)  # use fixed learning rate
		model.build_vocab(sentences.to_array())

		# model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)

		# model.build_vocab(sentences.to_array())

		for epoch in range(5):
			model.train(sentences.sentences_perm())

			# model.train(sentences)
			# model.alpha -= 0.002  # decrease the learning rate
			# model.min_alpha = model.alpha  # fix the learning rate, no decay

		model.save("../models/test_perm_10_epoch.d2v")

	print model.most_similar('0RjjaEK2565y_2014_29')
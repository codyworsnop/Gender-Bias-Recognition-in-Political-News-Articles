from __future__ import print_function, division
from matplotlib import pyplot as plt
import json
import numpy as np

import ApplicationConstants
from DataReader import DataReader
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
from nltk.tokenize import word_tokenize
import random 
import os.path

class doc():

	def Load_Model(self, article_doc2vec_model_path):
		if (os.path.exists(article_doc2vec_model_path)):
			model = Doc2Vec.load(article_doc2vec_model_path)
			return model 
		return None

	def Embed(self, articles, labels):

		tagged_doc_articles = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[labels[i]]) for i, _d in enumerate(articles)]
		#random.shuffle(tagged_doc_articles)

		#dm 1 is pv-dm, dm 0 is pv-dbow size is feature vec size, alpha is lr, negative is noise words, sample is thresh for down smample
		model = Doc2Vec(vector_size=50, alpha = 0.001, min_alpha = 0.00025, min_count = 1, epochs=100, negative=1, dm = 0, workers = multiprocessing.cpu_count()) 
		model.build_vocab(tagged_doc_articles)
		model.train(tagged_doc_articles, total_examples = model.corpus_count, epochs= model.epochs)

		return model

		#model.save("d2v.model")
		#print("model saved")
	
	def gen_vec(self, model, articles, labels):
		tagged_doc_articles = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[labels[i]]) for i, _d in enumerate(articles)]
		sents = tagged_doc_articles
		targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
		return targets, feature_vectors





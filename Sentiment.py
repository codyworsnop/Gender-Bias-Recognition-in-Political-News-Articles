#######This file calculates sentiment (OLD)#######
from DataReader import DataReader
from Models.SVM_engine import SVM
from Models.NN_engine import NN
import numpy as np 
from doc2vec import doc
import os
import ApplicationConstants
from imdb_data import LabeledLineSentence
from Visualizer import Visualizer
from Visualizer import GraphType
from sklearn.metrics import accuracy_score

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np 
import nltk
from doc2vec import doc
from DataReader import DataReader
import ApplicationConstants
from enum import Enum

from nltk.sentiment.vader import SentimentIntensityAnalyzer

class Sentiment():

	def __init__(self):
		self.SentAnalyzer = SentimentIntensityAnalyzer()
		self.docEmbed = doc()
		self.Visualizer = Visualizer() 

	def imdb(self, model, label_path, vector_path):
		sources = {'store/test-neg.txt':'TEST_NEG', 'store/test-pos.txt':'TEST_POS', 'store/train-neg.txt':'TRAIN_NEG', 'store/train-pos.txt':'TRAIN_POS' }
		sentences = LabeledLineSentence(sources)
		vectors, labels = sentences.generate_imdb_vec(model, label_path, vector_path)
		return vectors, labels

	def calculate_sent_score(self, article, polarities):
 	
		tokenizer = nltk.RegexpTokenizer(r"\w+")
		tokens = tokenizer.tokenize(article.Content) 
		avg_polarity = [] 	
		negations = ['not', 'isn\'t', 'wasn\'t', ]
		#word_polarities = dict(polarities)
		
		for index, token in enumerate(tokens):
			

			if (token in polarities):

				value = float(polarities[token])
				if (tokens[index - 1] in negations):
					value *= -1 

				avg_polarity.append(value)

		if len(avg_polarity) == 0:
			return 0.5
		
		return sum(avg_polarity) / len(avg_polarity)
	
	def train_sent_customlexicon(self, all_articles, leanings, reader):

		male = []
		female = [] 
		file = open('Data/polarities.txt', 'rb')
		polarities = pickle.load(file)
		file.close()
		#polarities = reader.load_politics('store/politics.tsv')

		for article_index, article in enumerate(all_articles):

			sentiment_score = self.calculate_sent_score(article, polarities)

			if (article.Label.TargetGender == ApplicationConstants.female_value):
				female.append((leanings[article_index], sentiment_score, None))
			else:
				male.append((leanings[article_index], sentiment_score, None))

		self.Visualizer.graph_sentiment(female, male, GraphType.StackedBargraph)
		
	def train_sent_vader(self, all_articles, leanings):

		male = []
		female = [] 

		for article_index, article in enumerate(all_articles):

			sentiment_score = self.SentAnalyzer.polarity_scores(article.Content)

			if (article.Label.TargetGender == ApplicationConstants.female_value):
				female.append((leanings[article_index], sentiment_score, None))
			else:
				male.append((leanings[article_index], sentiment_score, None))

		self.Visualizer.graph_sentiment(female, male, GraphType.StackedBargraph)

	def train_sent_models(self, all_articles, leanings, article_doc2vec_label_path, article_doc2vec_vector_path, article_doc2vec_model_path, imdb_label_path, imdb_vector_path):

		articles = list(map(lambda article: article.Content, all_articles))
		labels = list(map(lambda article: article.Label.TargetGender, all_articles))

		if (not os.path.exists(article_doc2vec_model_path)):
			all_articles_model = self.docEmbed.Embed(articles, labels) 
			all_articles_model.save(article_doc2vec_model_path)
		else:
			all_articles_model = self.docEmbed.Load_Model(article_doc2vec_model_path)

		if (not os.path.exists(article_doc2vec_label_path) or not os.path.exists(article_doc2vec_vector_path)):

			all_articles_labels, all_articles_vectors = self.docEmbed.gen_vec(all_articles_model, articles, labels)
			np.save(article_doc2vec_label_path, all_articles_labels)
			np.save(article_doc2vec_vector_path, all_articles_vectors)
			
		else:

			all_articles_labels = np.load(article_doc2vec_label_path)
			all_articles_vectors = np.load(article_doc2vec_vector_path)			

		#imdb_vec, imdb_labels = self.imdb(all_articles_model, imdb_label_path, imdb_vector_path)
		sources = {'store/test-neg.txt':'TEST_NEG', 'store/test-pos.txt':'TEST_POS', 'store/train-neg.txt':'TRAIN_NEG', 'store/train-pos.txt':'TRAIN_POS' }
		lls = LabeledLineSentence(sources)

		sentences = lls.to_array()
		words = list(map(lambda word: " ".join(word), list(map(lambda sentence: sentence.words, sentences))))
		labels = list(map(lambda label: " ".join(label), list(map(lambda sentence: sentence.tags, sentences))))

		for index, label in enumerate(labels): 
			if "NEG" in label: 
				labels[index] = 0
			elif "POS" in label:
				labels[index] = 1
		imdb_model = self.docEmbed.Embed(words, labels) 
		imdb_labels, imdb_vec = self.docEmbed.gen_vec(imdb_model, words, labels)

		models = [NN()]
		sents = [] 

		imdb_train = imdb_vec[0: int(len(imdb_vec) * 0.8)]
		imdb_train_label = imdb_labels[0: int(len(imdb_labels) * 0.8)]
		imdb_test = imdb_vec[int(len(imdb_vec) * 0.8): int(len(imdb_vec))]
		imdb_test_label = imdb_labels[int(len(imdb_labels) * 0.8): int(len(imdb_labels))]
		
		for model in models:

			male = []
			female = [] 
			model.Train(imdb_train, imdb_train_label, None, None)
			imdbPredictions = model.Predict(imdb_test)

			accuracy = accuracy_score(imdb_test_label, imdbPredictions) 
			print("ACCURACY:", accuracy)
			predictions = model.Predict(all_articles_vectors)
			
			for index, prediction in enumerate(predictions):

				sents.append((all_articles[index], labels[index], prediction, 0, leanings[index]))				  
				if int(labels[index]) == ApplicationConstants.female_value:
					female.append((leanings[index], prediction, 0))
				else:
					male.append((leanings[index], prediction, 0))

			#self.print_sents(sents)
			bFn, bFp, fFn, fFp, uFn, uFp, hFp, hFn, nFp, nFn = self.calc_plane_dist(female)
			bMn, bMp, fMn, fMp, uMn, uMp, hMp, hMn, nMp, nMn = self.calc_plane_dist(male)
			print("In order female pos/female neg/male pos/male neg")
			print("Breitbart: " + str(bFp) + " " + str(bFn) + " " + str(bMp) + " " + str(bMn))
			print("Fox: " + str(fFp) + " " + str(fFn) + " " + str(fMp) + " " + str(fMn))
			print("USA: " + str(uFp) + " " + str(uFn) + " " + str(uMp) + " " + str(uMn))
			print("Huffpost: " + str(hFp) + " " + str(hFn) + " "+  str(hMp) + " " + str(hMn))
			print("NYT: " + str(nFp) + " " + str(nFn) + " " + str(nMp) + " " + str(nMn))

			self.Visualizer.graph_sentiment(female, male, GraphType.Line)

	
	def calc_plane_dist(self, gender):
		ttlBP = 0
		ttlBN = 0 
		bn = 0
		bp = 0
		ttlFP = 0
		ttlFN = 0
		fn = 0
		fp = 0
		ttlUP = 0
		ttlUN = 0
		un = 0
		up = 0
		ttlHP = 0
		ttlHN = 0
		hn = 0
		hp = 0
		ttlNP = 0
		ttlNN = 0
		nn = 0
		np = 0

		for i in range(len(gender)):
			if gender[i][0] == 'breitbart':
			
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlBN +=1
						bn += gender[i][2]
					else:
						ttlBP +=1
						bp += gender[i][2]
			if gender[i][0] == 'fox' :
				
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlFN +=1
						fn += gender[i][2]
					else:
						ttlFP +=1
						fp += gender[i][2]
			if gender[i][0] == 'usa_today' :
				
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlUN +=1
						un += gender[i][2]
					else:
						ttlUP +=1
						up += gender[i][2]
			if gender[i][0] == 'huffpost' :
				
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlHN +=1
						hn += gender[i][2]
					else:
						ttlHP +=1
						hp += gender[i][2]
			if gender[i][0] == 'new_york_times' :
				
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlNN +=1
						nn += gender[i][2]
					else:
						ttlNP +=1
						np += gender[i][2]

		if bn != 0:
			bn = bn/ttlBN
		else:
			bn = 0
		if bp != 0:
			bp = bp /ttlBP
		else:
			bp = 0
		if fn != 0:
			fn = fn/ttlFN
		else:
			fn = 0
		if fp != 0:
			fp = fp /ttlFP
		else:
			fp = 0
		if un != 0:
			un = un/ttlUN
		else:
			un = 0
		if up != 0:
			up = up /ttlUP
		else:
			up = 0
		if hn != 0:
			hn = hn/ttlHN
		else:
			hn = 0
		if hp != 0:
			hp = hp /ttlHP
		else:
			hp = 0
		if nn != 0:
			nn = nn/ttlNN
		else:
			nn = 0
		if np != 0:
			np = np /ttlNP
		else:
			np = 0
		


		return bn, bp, fn, fp, un, up, hn, hp, nn, np


	def print_sents(self, sents):

		for sent in sents:

			print('\n')

			print('leaning', sent[4])
			#print("content:", sent[0].Content)
			print("target:", sent[0].Label.TargetName)
			print("prediction", sent[2])
			print("confidence", sent[3])

if __name__ == "__main__":

	sentiment = Sentiment()
	reader = DataReader()

	splits = reader.Load_Splits(ApplicationConstants.all_articles_random_v2, None, number_of_articles=50, clean=False, save=False, shouldRandomize=False)
  
	leanings_articles = list(map(lambda leaning: splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test], splits[0]))
	
	leanings = [] #flattened leanings

	for leaning in splits[0]:
		for article in range(len(splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test])):
			leanings.append(leaning)

	articles = [item for sublist in leanings_articles for item in sublist]
	#sentiment.train_sent_vader(articles, leanings)

	sentiment.train_sent_customlexicon(articles, leanings, reader)

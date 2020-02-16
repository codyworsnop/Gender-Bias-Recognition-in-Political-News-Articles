#classes

from DataReader import DataReader
from DataContracts import Article
from doc2vec import doc
from SentimentIntensityAnalyzer import SentimentAnalyzer
from Metrics import Metrics
from Visualizer import Visualizer 
from imdb_data import LabeledLineSentence
import ApplicationConstants

#models
from Models.SVM_engine import SVM
from Models.KNN_engine import KNN
from Models.Naive_Bayes_engine import Naive_Bayes
from Models.Linear_Classification_engine import Linear_Classifier 
from Models.NN_engine import NN

#helpers
import statistics
import numpy as np 
import matplotlib.pyplot as plt 
import os.path
from nltk.tokenize import sent_tokenize

import sys

class Orchestrator():

	def __init__(self):
		self.Reader = DataReader()
		
		self.Splits = None 
		self.Sources = None
		self.docEmbed = doc()
		self.Metrics = Metrics()
		self.Visualizer = Visualizer() 
		self.SentimentAnalyzer = SentimentAnalyzer() 

	def read_data(self, path, clean=True, save=False, number_of_articles = 50, splits=True, neutral=False):       
		return self.Reader.Load_Splits(path, clean=clean, save=save, number_of_articles=number_of_articles, split=splits, neutral=neutral)
	
	def get_count(self, articles) -> int:
		article_count = 0 

		for article in articles:
			tokens = sent_tokenize(article)
			article_count += len(tokens)

		return article_count


	def imdb(self, model, label_path, vector_path):
		sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS' }
		sentences = LabeledLineSentence(sources)
		vectors, labels = sentences.generate_imdb_vec(model, label_path, vector_path)
		return vectors, labels

	def test_sent_models(self, all_articles, leanings, article_doc2vec_model_path, imdb_label_path, imdb_vector_path):

		model = self.docEmbed.Load_Model(article_doc2vec_model_path) 

		#need empty labels for the tagged document 
		neutral_labels = [0] * len(neutral_articles)
		neutral_labels, neutral_vectors = self.docEmbed.gen_vec(model, all_articles, neutral_labels)

		imdb_vec, imdb_labels = self.imdb(model, imdb_label_path, imdb_vector_path)
		models = [SVM()]
		for model in models:

			sentiments = [] 

			model.Train(imdb_vec, imdb_labels, None, None)
			predictions, _ = model.Predict(neutral_vectors)

			for index, prediction in enumerate(predictions):

					sentiments.append((leanings[index], prediction))

			self.Visualizer.graph_neutral(sentiments)


	def train_sent_models(self, all_articles, all_labels, leanings, article_doc2vec_label_path, article_doc2vec_vector_path, article_doc2vec_model_path, imdb_label_path, imdb_vector_path):
		if (not os.path.exists(article_doc2vec_model_path)):
			all_articles_model = self.docEmbed.Embed(all_articles, all_labels) 
			all_articles_model.save(article_doc2vec_model_path)
		else:
			all_articles_model = self.docEmbed.Load_Model(article_doc2vec_model_path) 

		if (not os.path.exists(article_doc2vec_label_path) or not os.path.exists(article_doc2vec_vector_path)):

			all_articles_labels, all_articles_vectors = self.docEmbed.gen_vec(all_articles_model, all_articles, all_labels)

			np.save(article_doc2vec_label_path, all_articles_labels)
			np.save(article_doc2vec_vector_path, all_articles_vectors)
			
		else:

			all_articles_labels = np.load(article_doc2vec_label_path)
			all_articles_vectors = np.load(article_doc2vec_vector_path)
				
		# for index, label in enumerate(all_articles_labels):
		# 	all_articles_labels[index] = (label, leanings[index])

		imdb_vec, imdb_labels = self.imdb(all_articles_model, imdb_label_path, imdb_vector_path)

		models = [SVM()]

		for model in models:

			male = []
			female = [] 

			model.Train(imdb_vec, imdb_labels, None, None)
			predictions, confidence = model.Predict(all_articles_vectors)
			
			for index, prediction in enumerate(predictions):
									  
				if int(all_articles_labels[index]) == ApplicationConstants.female_value:
					female.append((leanings[index], prediction))
				else:
					male.append((leanings[index], prediction))

			self.Visualizer.graph_sentiment(female, male)

	#def print_shit(self, fileName, allF, allM, conf):
	def print_shit(self, fileName, allF, allM):
		file = open(fileName, 'w')
		print('FEMALE\n', file = file)
		print(allF, file = file)
		print('\nMALE\n',  file = file)
		print(allM,  file = file)
		#print('\nPROBABILITIES\n',  file = file)
		#print(conf, file=file)


	def embed_fold(self, articles, labels, fold, leaning):
		''' 
		trains and returns the vector embeddings for doc2vec or sent2vec 

		Parameters:5
		articles: a list of articles that are cleaned
		labels: a list of labels corresponding to the article genders
		''' 

		#emb = self.docEmbed.word2vec() 
		targets, regressors, model = self.docEmbed.Embed(articles, labels)

		return list(targets), regressors, model
	
	def train_all(self, splits):
		''' trains all models against all leanings
		
		Parameters: 
		------------
		splits: A list of the splits 

		''' 
		models = [SVM(), KNN(), Naive_Bayes(), Linear_Classifier(), NN()]

		#models = [NN()]

		split_count = 0 

		#for each split
		for split in splits:
			
			print("Starting split:", str(split_count), "\n")
			split_count += 1

			#loop over all leanings
			for leaning in split:

				print("For leaning:", leaning.upper())
				
				#train embeddings
				training_dataset = split[leaning][ApplicationConstants.Train]

				#validation embeddings 
				validation_dataset = split[leaning][ApplicationConstants.Validation]

				#test embeddings
				test_dataset = split[leaning][ApplicationConstants.Test]           

				article_labels, article_embeddings, article_model = self.embed_fold(list(map(lambda article: article.Content, training_dataset + validation_dataset + test_dataset)), list(map(lambda article: article.Label.TargetGender, training_dataset + validation_dataset + test_dataset)), split_count, leaning)
				training_embeddings = article_embeddings[:len(training_dataset)]
				training_labels = article_labels[:len(training_dataset)]
				validation_embeddings = article_embeddings[len(training_dataset): len(training_dataset) + len(validation_dataset)]
				validation_labels = article_labels[len(training_dataset): len(training_dataset) + len(validation_dataset)]

				test_embeddings = article_embeddings[len(training_dataset) + len(validation_dataset):]
				test_labels = article_labels[len(training_dataset) + len(validation_dataset):]

				for model in models: 

					#get prediction from embeddings 
					model.Train(training_embeddings, training_labels, validation_embeddings, validation_labels)
					prediction, confidence = model.Predict(test_embeddings)


				#model = models[0] 
				#model.Model.coefs_[model.Model.n_layers_ - 2]
				if split_count == 1:
					self.Visualizer.plot_TSNE(leaning, training_embeddings + validation_embeddings + test_embeddings, training_labels + validation_labels + test_labels, training_dataset + validation_dataset + test_dataset)
		
	  
orchestrator = Orchestrator()
#splits = orchestrator.read_data(ApplicationConstants.all_articles, clean=False, save=False, number_of_articles=25) 
cleaned_splits = orchestrator.read_data(ApplicationConstants.cleaned_news_root_path, clean=False, save=False, number_of_articles=50)
neutral_data = orchestrator.read_data(ApplicationConstants.articles_neutral, clean=True, save=False, number_of_articles=20, splits=False, neutral=True)

#train embeddings - uncleaned 
# leanings_articles = list(map(lambda leaning: splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test], splits[0]))
# count = len(leanings_articles)
# leanings = []

# for leaning in splits[0]:
# 	for article in range(len(splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test])):
# 		leanings.append(leaning) 

# flat_list = [item for sublist in leanings_articles for item in sublist]

# articles = list(map(lambda article: article.Content, flat_list))  
# labels = list(map(lambda article: article.Label.TargetGender, flat_list))
			

#train embeddings - cleaned 
leanings_articles = list(map(lambda leaning: cleaned_splits[0][leaning][ApplicationConstants.Train] + cleaned_splits[0][leaning][ApplicationConstants.Validation] + cleaned_splits[0][leaning][ApplicationConstants.Test], cleaned_splits[0]))
leanings = []

for leaning in cleaned_splits[0]:
	for article in range(len(cleaned_splits[0][leaning][ApplicationConstants.Train] + cleaned_splits[0][leaning][ApplicationConstants.Validation] + cleaned_splits[0][leaning][ApplicationConstants.Test])):
		leanings.append(leaning) 

flat_list = [item for sublist in leanings_articles for item in sublist]
cleaned_articles = list(map(lambda article: article.Content, flat_list))  
cleaned_labels = list(map(lambda article: article.Label.TargetGender, flat_list))

orchestrator.train_sent_models(cleaned_articles, cleaned_labels, leanings, ApplicationConstants.all_articles_doc2vec_labels_cleaned_path, ApplicationConstants.all_articles_doc2vec_vector_cleaned_path, ApplicationConstants.all_articles_doc2vec_model_cleaned_path, ApplicationConstants.imdb_sentiment_label_cleaned_path, ApplicationConstants.imdb_sentiment_vector_cleaned_path)

#neutral data
leanings_articles = list(map(lambda data: data[1], neutral_data))
neutral_leanings = []
for leaning in neutral_data:
	for article in range(len(leaning[1])):
		neutral_leanings.append(leaning[0]) 
flat_list = [item for sublist in leanings_articles for item in sublist]
neutral_articles = list(map(lambda article: article.Content, flat_list))  


orchestrator.test_sent_models(neutral_articles, neutral_leanings, ApplicationConstants.all_articles_doc2vec_model_cleaned_path, ApplicationConstants.imdb_sentiment_label_cleaned_path, ApplicationConstants.imdb_sentiment_vector_cleaned_path)






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


class Orchestrator():

	def __init__(self):
		self.Reader = DataReader()
		
		self.Splits = None 
		self.Sources = None
		self.docEmbed = doc()
		self.Metrics = Metrics()
		self.Visualizer = Visualizer() 
		self.SentimentAnalyzer = SentimentAnalyzer() 

	def read_data(self, path, savePath=None, clean=True, save=False, random=False, number_of_articles = 50):       
		return self.Reader.Load_Splits(path, savePath=savePath, clean=clean, save=save, shouldRandomize=random, number_of_articles=number_of_articles)

	def imdb(self, model, label_path, vector_path):
		sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS' }
		sentences = LabeledLineSentence(sources)
		vectors, labels = sentences.generate_imdb_vec(model, label_path, vector_path)
		return vectors, labels

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

		imdb_vec, imdb_labels = self.imdb(all_articles_model, imdb_label_path, imdb_vector_path)

		models = [SVM()]
		sents = [] 

		for model in models:

			male = []
			female = [] 

			model.Train(imdb_vec, imdb_labels, None, None)
			predictions, confidences = model.Predict(all_articles_vectors)
			
			for index, prediction in enumerate(predictions):

				sents.append((all_articles[index], labels[index], prediction, confidences[index], leanings[index]))				  
				if int(labels[index]) == ApplicationConstants.female_value:
					female.append((leanings[index], prediction, confidences[index]))
				else:
					male.append((leanings[index], prediction, confidences[index]))

			self.print_sents(sents)
			self.Visualizer.graph_sentiment(female, male)

	
			bFn, bFp, fFn, fFp, uFn, uFp, hFp, hFn, nFp, nFn = self.calc_plane_dist(female)
			bMn, bMp, fMn, fMp, uMn, uMp, hMp, hMn, nMp, nMn = self.calc_plane_dist(male)
			print("In order female pos/female neg/male pos/male neg")
			print("Breitbart: " + str(bFp) + " " + str(bFn) + " " + str(bMp) + " " + str(bMn))
			print("Fox: " + str(fFp) + " " + str(fFn) + " " + str(fMp) + " " + str(fMn))
			print("USA: " + str(uFp) + " " + str(uFn) + " " + str(uMp) + " " + str(uMn))
			print("Huffpost: " + str(hFp) + " " + str(hFn) + " "+  str(hMp) + " " + str(hMn))
			print("NYT: " + str(nFp) + " " + str(nFn) + " " + str(nMp) + " " + str(nMn))

			self.Visualizer.graph_sentiment(female, male)

	def print_sents(self, sents):

		for sent in sents:

			print('\n')

			print('leaning', sent[4])
			print("content:", sent[0].Content)
			print("target:", sent[0].Label.TargetName)
			print("prediction", sent[2])
			print("confidence", sent[3])

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
		model = self.docEmbed.Embed(articles, labels)
		targets, regressors = self.docEmbed.gen_vec(model, articles, labels)

		return list(targets), regressors, model
	
	def train_all(self, splits):
		''' trains all models against all leanings
		
		Parameters: 
		------------
		splits: A list of the splits 

		''' 
		models = [SVM(), KNN(), Naive_Bayes(), Linear_Classifier(), NN()]
		bP = []
		bR = []
		bF = []
		fP = []
		fR = []
		fF = []
		uP = []
		uR = []
		uF = []
		hP = []
		hR = []
		hF = []
		nP = []
		nR = []
		nF = []

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
					
					print("Model:", str(type(model)).split('.')[2].split('\'')[0], "precision:", self.Metrics.Precision(prediction, test_labels), "recall:", self.Metrics.Recall(prediction, test_labels), "F-Measure:", self.Metrics.Fmeasure(prediction, test_labels))   

				
				#model = models[0] 
				#model.Model.coefs_[model.Model.n_layers_ - 2]
				if split_count == 1:
					self.Visualizer.plot_TSNE(leaning, training_embeddings + validation_embeddings + test_embeddings, training_labels + validation_labels + test_labels, training_dataset + validation_dataset + test_dataset)
		

	def get_most_sig_sent(self, articles, context_sentence_number = 2):

		for article_index, article in enumerate(articles): 

			target = article.Label.TargetName
			lastname = target.split('_')[1]
			qualtive_sentences = []
			sentences = article.Content.split('.')

			#start 2 over so we have context
			for sentence_index in range(2, len(sentences)):

				#check if name in sentence
				if lastname in sentences[sentence_index]:
					
					for context_number in range(-context_sentence_number, context_sentence_number):

						if context_number + sentence_index < len(sentences):
							qualtive_sentences.append(sentences[sentence_index + context_number])

					break 

			articles[article_index].Content = " ".join(qualtive_sentences)

		return articles

	def calc_metrics(self, bP, fP, uP, hP, nP ):
		BttlS = 0
		BttlK = 0
		BttlN = 0
		BttlL = 0
		BttlNet = 0
		FttlS = 0
		FttlK = 0
		FttlN = 0
		FttlL = 0
		FttlNet = 0
		UttlS = 0
		UttlK = 0
		UttlN = 0
		UttlL = 0
		UttlNet = 0
		HttlS = 0
		HttlK = 0
		HttlN = 0
		HttlL = 0
		HttlNet = 0
		NttlS = 0
		NttlK = 0
		NttlN = 0
		NttlL = 0
		NttlNet = 0

		
		for i in range(len(bP)):
			if i %5 == 0:
				BttlS +=bP[i]
				FttlS +=fP[i]
				UttlS += uP[i]
				HttlS += hP[i]
				NttlS += nP[i]
			if i % 5 == 1:
				BttlK +=bP[i]
				FttlK +=fP[i]
				UttlK += uP[i]
				HttlK += hP[i]
				NttlK += nP[i]
			if i % 5 == 2:
				BttlN +=bP[i]
				FttlN +=fP[i]
				UttlN += uP[i]
				HttlN += hP[i]
				NttlN += nP[i]
			if i %5 == 3:
				BttlL +=bP[i]
				FttlL +=fP[i]
				UttlL += uP[i]
				HttlL += hP[i]
				NttlL += nP[i]
			if i%5 == 4:
				BttlNet +=bP[i]
				FttlNet +=fP[i]
				UttlNet += uP[i]
				HttlNet += hP[i]
				NttlNet += nP[i]
		bp = bP
		print("Breitbart SVM: " + str(BttlS /(len(bP)/5)) + " Breitbart KNN: " + str(BttlK/(len(bP)/5)) + " Breitbart NB: " + str(BttlN /(len(bP)/5)) + " Breitbart LC: " +str(BttlL /(len(bP)/5)) + " Breitbart NN: " + str(BttlNet/(len(bP)/5)))
		print("Fox SVM: " + str(FttlS /(len(bP)/5)) + " Fox KNN: " + str(FttlK/(len(bP)/5)) + " Fox NB: " + str(FttlN /(len(bP)/5)) + " Fox LC: " +str(FttlL /(len(bP)/5)) + " Fox NN: " + str(FttlNet/(len(bP)/5)))
		print("USA SVM: " + str(UttlS /(len(bP)/5)) + " USA KNN: " + str(UttlK/(len(bP)/5)) + " USA NB: " + str(UttlN /(len(bP)/5)) + " USA LC: " +str(UttlL /(len(bP)/5)) + " USA NN: " + str(UttlNet/(len(bP)/5)))
		print("Huffpost SVM: " + str(HttlS /(len(bP)/5)) + " Huffpost KNN: " + str(HttlK/(len(bp)/5)) + " Huffpost NB: " + str(HttlN /(len(bp)/5)) + " Huffpost LC: " +str(HttlL /(len(bp)/5)) + " Huffpost NN: " + str(HttlNet/(len(bp)/5)))
		print("NYT SVM: " + str(NttlS /(len(bP)/5)) + " NYT KNN: " + str(NttlK/(len(bp)/5)) + " NYT NB: " + str(NttlN /(len(bp)/5)) + " NYT LC: " +str(NttlL /(len(bp)/5)) + " NYT NN: " + str(NttlNet/(len(bp)/5)))
	

orchestrator = Orchestrator()
splits = orchestrator.read_data(ApplicationConstants.all_articles_random, clean=False, save=False, number_of_articles=1000) 

leanings_articles = list(map(lambda leaning: splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test], splits[0]))
leanings = []

for leaning in splits[0]:
	for article in range(len(splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test])):
 		leanings.append(leaning) 

articles = [item for sublist in leanings_articles for item in sublist]
orchestrator.train_sent_models(articles, leanings, ApplicationConstants.all_articles_doc2vec_labels_cleaned_path, ApplicationConstants.all_articles_doc2vec_vector_cleaned_path, ApplicationConstants.all_articles_doc2vec_model_cleaned_path, ApplicationConstants.imdb_sentiment_label_cleaned_path, ApplicationConstants.imdb_sentiment_vector_cleaned_path)

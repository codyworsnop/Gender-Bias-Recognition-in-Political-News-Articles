#######This file runs several processes #######
#classes


from DataReader import DataReader
from DataContracts import Article
from doc2vec import doc
from Metrics import Metrics
from Visualizer import Visualizer
from imdb_data import LabeledLineSentence
import ApplicationConstants
import StopWords
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#models
from Models.SVM_engine import SVM
from Models.KNN_engine import KNN
from Models.Naive_Bayes_engine import Naive_Bayes
from Models.Linear_Classification_engine import Linear_Classifier
from Models.NN_engine import NN
from Models.NN_engine import  Linear_NN


#helpers
import statistics
import numpy as np 
import matplotlib.pyplot as plt
import os.path
import timeit
import spacy
import nltk
import re
import random
import pickle

class Orchestrator():

	def __init__(self):
		self.Reader = DataReader()
		
		self.Splits = None 
		self.Sources = None
		self.docEmbed = doc()
		self.Metrics = Metrics()
		self.Visualizer = Visualizer()

	def read_data(self, path, savePath=None, clean=True, save=False, random=False, number_of_articles = 50, pos_tagged = False):
		return self.Reader.Load_Splits(path, savePath=savePath, clean=clean, save=save, shouldRandomize=random, number_of_articles=number_of_articles, pos_tagged = pos_tagged)

	def read_data_csv(self, path, savePath=None, clean=True, save=False, random=False, number_of_articles = 50):
		return self.Reader.Load_ATN_csv(path, savePath=savePath, clean=clean, save=save, shouldRandomize=random, number_of_articles=number_of_articles)

	def imdb(self, model, label_path, vector_path):
		sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS' }
		sentences = LabeledLineSentence(sources)
		vectors, labels = sentences.generate_imdb_vec(model, label_path, vector_path)
		return vectors, labels

	

	def print_sents(self, sents):

		for sent in sents:

			print('\n')

			print('leaning', sent[4])
			print("content:", sent[0].Content)
			print("target:", sent[0].Label.TargetName)
			print("prediction", sent[2])
			print("confidence", sent[3])

	def calc_plane_dist(self, gender):
		ttlBreitbartPos,ttlBreitbartNeg, breitbartneg, breitbartpos, ttlFoxPos, ttlFoxNeg, foxneg, foxpos, ttlUsaPos, ttlUsaNeg, usaneg, usapos, ttlHuffPos, ttlHufNeg, huffneg, huffpos, ttlNytPos, ttlNytNeg, nytneg, nytpos = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

		for i in range(len(gender)):
			if gender[i][0] == 'breitbart':
			
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlBreitbartNeg +=1
						breitbartneg += gender[i][2]
					else:
						ttlBreitbartPos +=1
						breitbartpos += gender[i][2]
			if gender[i][0] == 'fox' :
				
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlFoxNeg+=1
						foxneg += gender[i][2]
					else:
						ttlFoxPos +=1
						foxpos += gender[i][2]
			if gender[i][0] == 'usa_today' :
				
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlUsaNeg +=1
						usaneg += gender[i][2]
					else:
						ttlUsaPos +=1
						usapos += gender[i][2]
			if gender[i][0] == 'huffpost' :
				
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlHufNeg+=1
						huffneg += gender[i][2]
					else:
						ttlHuffPos +=1
						huffpos += gender[i][2]
			if gender[i][0] == 'new_york_times' :
				
				if abs(gender[i][2]) >= 0.25:
					
					if gender[i][2] < 0:
						ttlNytNeg +=1
						nytneg += gender[i][2]
					else:
						ttlNytPos +=1
						nytpos += gender[i][2]

		if breitbartneg != 0:
			breitbartneg = breitbartneg/ttlBreitbartNeg
		else:
			breitbartneg = 0
		if breitbartpos != 0:
			breitbartpos = breitbartpos/ttlBreitbartPos
		else:
			breitbartpos = 0
		if foxneg != 0:
			foxneg = foxneg/ttlFoxNeg
		else:
			foxneg = 0
		if foxpos != 0:
			foxpos = foxpos /ttlFoxPos
		else:
			foxpos = 0
		if usaneg != 0:
			usaneg = usaneg/ttlUsaNeg
		else:
			usaneg = 0
		if usapos != 0:
			usapos = usapos /ttlUsaPos
		else:
			usapos = 0
		if huffneg != 0:
			huffneg = huffneg/ttlHufNeg
		else:
			huffneg = 0
		if huffpos != 0:
			huffpos = huffpos /ttlHuffPos
		else:
			huffpos = 0
		if nytneg != 0:
			nytneg = nytneg/ttlNytNeg
		else:
			nytneg = 0
		if nytpos != 0:
			nytpos = nytpos /ttlNytPos
		else:
			np = 0
		


		return breitbartneg, breitbartpos, foxneg, foxpos, usaneg, usapos, huffneg, huffpos, nytneg, nytpos



	def print(self, fileName, allF, allM):
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

		model = self.docEmbed.Embed(articles, labels)
		targets, regressors = self.docEmbed.gen_vec(model, articles, labels)

		return list(targets), regressors, model
	
	def train_all(self, splits):
		''' trains all models against all leanings
		
		Parameters: 
		------------
		splits: A list of the splits 

		''' 
		#models = [SVM(), KNN(), Naive_Bayes(), Linear_Classifier(), NN()]
		models = [NN()]
		bP, bR, bF, fP, fR, fF, uP, uR, uF, hP, hR, hF, nP, nR, nF = [], [], [], [], [], [], [], [], [], [], [], [], [], []



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
		BreitbartTtlSvm, BreitbartTtlKnn, BreitbartTtlNB, BreitbartTtlLin, BreitbartTtlNN, FoxTtlSvm, FoxTtlKnn, FoxTtlNB, FoxTtlLin, FoxTtlNN, UsaTtlSvm, UsaTtlKnn, UsaTtlNB, UsaTtlLin, UsaTtlNN  = 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0
		HuffTtlSvm, HuffTtlKnn, HuffTtlNB, HuffTtlLin, HuffTtlNN, NytTtlSvm, NytTtlKnn, NytTtlNB, NytTtlLin, NytTtlNN = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0


		for i in range(len(bP)):
			if i %5 == 0:
				BreitbartTtlSvm +=bP[i]
				FoxTtlSvm +=fP[i]
				UsaTtlSvm += uP[i]
				HuffTtlSvm += hP[i]
				NytTtlSvm += nP[i]
			if i % 5 == 1:
				BreitbartTtlKnn +=bP[i]
				FoxTtlKnn +=fP[i]
				UsaTtlKnn += uP[i]
				HuffTtlKnn += hP[i]
				NytTtlKnn += nP[i]
			if i % 5 == 2:
				BreitbartTtlNB +=bP[i]
				FoxTtlNB +=fP[i]
				UsaTtlNB += uP[i]
				HuffTtlNB += hP[i]
				NytTtlNB += nP[i]
			if i %5 == 3:
				BreitbartTtlLin +=bP[i]
				FoxTtlLin +=fP[i]
				UsaTtlLin += uP[i]
				HuffTtlLin += hP[i]
				NytTtlLin += nP[i]
			if i%5 == 4:
				BreitbartTtlNN +=bP[i]
				FoxTtlNN +=fP[i]
				UsaTtlNN += uP[i]
				HuffTtlNN += hP[i]
				NytTtlNN += nP[i]
		bp = bP
		print("Breitbart SVM: " + str(BreitbartTtlSvm /(len(bP)/5)) + " Breitbart KNN: " + str(BreitbartTtlKnn/(len(bP)/5)) + " Breitbart NB: " + str(BreitbartTtlNB /(len(bP)/5)) + " Breitbart LC: " +str(BreitbartTtlLin /(len(bP)/5)) + " Breitbart NN: " + str(BreitbartTtlNN/(len(bP)/5)))
		print("Fox SVM: " + str(FoxTtlSvm /(len(bP)/5)) + " Fox KNN: " + str(FoxTtlKnn/(len(bP)/5)) + " Fox NB: " + str(FoxTtlNB /(len(bP)/5)) + " Fox LC: " +str(FoxTtlLin /(len(bP)/5)) + " Fox NN: " + str(FoxTtlNN/(len(bP)/5)))
		print("USA SVM: " + str(UsaTtlSvm/(len(bP)/5)) + " USA KNN: " + str(UsaTtlKnn/(len(bP)/5)) + " USA NB: " + str(UsaTtlNB /(len(bP)/5)) + " USA LC: " +str(UsaTtlLin /(len(bP)/5)) + " USA NN: " + str(UsaTtlNN/(len(bP)/5)))
		print("Huffpost SVM: " + str(HuffTtlSvm /(len(bP)/5)) + " Huffpost KNN: " + str(HuffTtlKnn/(len(bp)/5)) + " Huffpost NB: " + str(HuffTtlNB /(len(bp)/5)) + " Huffpost LC: " +str(HuffTtlLin /(len(bp)/5)) + " Huffpost NN: " + str(HuffTtlNN/(len(bp)/5)))
		print("NYT SVM: " + str(NytTtlSvm /(len(bP)/5)) + " NYT KNN: " + str(NytTtlKnn/(len(bp)/5)) + " NYT NB: " + str(NytTtlNB /(len(bp)/5)) + " NYT LC: " +str(NytTtlLin /(len(bp)/5)) + " NYT NN: " + str(NytTtlNN/(len(bp)/5)))

	def check_word_content(self, word_list, all_articles):
		articles = list(map(lambda article: article.Content, all_articles))

		bad_words = []
		ttl = 0
		#count = 0
		affected = []
		count_list = []
		print(len(articles))
		for i, article in enumerate(articles):
			for word in word_list:
				if re.search(rf'\b{word}\b' ,  article):
					if word not in bad_words:
						bad_words.append(word)
						count_list.append(1)

					else:
						ind = bad_words.index(word)
						val = count_list[ind]

						num_exist = val + article.count(word)
						count_list[ind] = num_exist

					affected.append(i)

		a = []
		print(affected)
		[a.append(x) for x in affected if x not in a]
		print(a)
		print(bad_words)
		print("total articles: ", len(a))
		zipped = list(zip(bad_words, count_list))
		print("total use: ",zipped)


	def calc_word_vector(self, all_articles, not_pos = True, lemmad = True, print_vocab=False):
		nlp = spacy.load("en_core_web_lg")
		word_vector = set([])
		punctuation = [',', '.', '\"', '"', '!', '?', '\'', '$', ''', '\n', '_', ':', ';', '%', '—', '–', ''','~','―','′', ',', '≠', '|',
					   '•', ' ', ', ', '/', '>', '<', '=', '-', '’', ']', '[', '(', ')', '{', '}', '@', '#', '^', '*', '&', '­']
		if not_pos:
			from nltk.corpus import stopwords
			stops = list(stopwords.words('english'))
		for i, split in enumerate(all_articles):
			print("Fold " + str(i + 1))
			for j, leaning in enumerate(split):
				print("calc word vec", leaning)
				training_dataset = split[leaning][ApplicationConstants.Train]
				validation_dataset = split[leaning][ApplicationConstants.Validation]
				test_dataset = split[leaning][ApplicationConstants.Test]
				all_articles = list(map(lambda art: art.Content, training_dataset + validation_dataset + test_dataset))
				for article in all_articles:
					document = nlp(article)
					if not_pos:
						for token in document:
							if not token.is_punct:
								if not lemmad:
									word = token.orth_.lower()
								else:
									word = token.lemma_.lower()
								for i, char in enumerate(word):
									if len(word) > 1:
										if word[0] in punctuation:
											word = word[1:]
										if word[-1] in punctuation:
											word = word[:-1]
								for punct in punctuation:
									if punct in word:
										word = '.'
								if "gpe" in word:
									word = "gpe"
								if "norp" in word:
									word = "norp"
								if word not in punctuation and word not in word_vector and word not in stops:
									if len(word) >= 2 and word != "\n" and ":" not in word:
										word_vector.add(word)

					else:
						#ms = [' M. ', ' m. ']
						for token in document:
							if token.pos_ is "ADJ" and token.orth_.lower() not in word_vector and token.text not in punctuation:# and token.text not in ms:
								if not lemmad:
									word = token.orth_.lower()
								else:
									word = token.lemma_.lower()
								for i, char in enumerate(word):
									if len(word) > 1:
										if word[0] in punctuation:
											word = word[1:]

										if word[-1] in punctuation:
											word = word[:-1]

								for punct in punctuation:
									if punct in word:
										word = '.'
								if "gpe" in word and "gpe" not in word_vector:
									word = "gpe"
								else:
									break
								if "norp" in word and "norp" not in word_vector:
									word = "norp"
								else:
									break
								if len(word) >= 2 and word != "\n" and ":" not in word:
									word_vector.add(word)

			if print_vocab:
				printed_word_vec = sorted(word_vector)
				if not_pos and lemmad:
					name = "vocabulary/fullVocab_lemmad.txt"
				elif not_pos and not lemmad:
					name = "vocabulary/fullVocab_notLemmad.txt"
				elif not not_pos and lemmad:
					name = "vocabulary/adjVocab_lemmad.txt"
				else:
					name = "vocabulary/adjVocab_notLammad.txt"
				fout = open(name, 'w')
				for item in printed_word_vec:
					fout.write(item + '\n')
				fout.close()

			return list(word_vector)

	def calc_count_doc_count_vector(self, word_vector, article, nlp, lemmad = False):

		words = nlp(article)
		count_vector = [0]*len(word_vector)
		count_set = {}
		punctuation = [',', '.', '\"', '"', '!', '?', '\'', '$', ''', '\n', '_', ':', ';', '%', '—', '–', ''',
					   '•', ' ', ', ', '/', '>', '<', '=', '-', '’', ']', '[', '(', ')', '{', '}', '@', '#', '^', '*',
					   '&', ':']
		for word in word_vector:
			count_set[word] = 0

		for token in words:
			if lemmad:
				word = token.lemma_.lower()
			else:
				word = token.orth_.lower()

			for i, char in enumerate(word):
				if len(word) > 1:
					if word[0] in punctuation:
						word = word[1:]

					if word[-1] in punctuation:
						word = word[:-1]

			for punct in punctuation:
				if punct in word:
					word = 'aklfjakldfjlaskf' #if there's punctuation still in the middle of the word, it's a garbage word, and we insert a garbage word that doesn't exist in the cum_vec
			if "gpe" in word:
				word = "gpe"
			if "norp" in word:
				word = "norp"
			if word in count_set:
				count_set[word] +=1

		for i in range(len(word_vector)):
			count_vector[i] = count_set[word_vector[i]]
		return count_vector

	def get_all_articles(self):
		splits = self.read_data(ApplicationConstants.all_articles_random_v2_cleaned, clean=True, save=True,
										savePath="./Data/articles_random_v2_cleaned.json",
										number_of_articles=50)  # article objects

		leanings_articles = list(map(
			lambda leaning: splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][
				ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test], splits[0]))
		# print(leanings_articles)
		leanings = []  # flattened leanings

		for leaning in splits[0]:
			for article in range(len(splits[0][leaning][ApplicationConstants.Train] + splits[0][leaning][
				ApplicationConstants.Validation] + splits[0][leaning][ApplicationConstants.Test])):
				leanings.append(leaning)
		articles = [item for sublist in leanings_articles for item in sublist]
		return articles

	def load_exisiting_bow(self, model, file_name_2):
		net = pickle.load(open(model, 'rb'))
		weights = net.Get_Weights()

		numpy_cumulative = np.load(file_name_2)
		count_vectors = numpy_cumulative.tolist()
		label_name = file_name_2[-4] + "_labels.npy"
		list_labels = np.load(label_name)
		list_labels = list_labels.tolist()
		trainLen = int(len(count_vectors) * 0.8)

		predictions = net.Predict(count_vectors[trainLen:])

		acc = accuracy_score(list_labels[trainLen:], predictions)
		target_names = ['Female', 'Male']
		print("accuracy is: " + str(acc))


	def run_bow(self, file_name_1, file_name_2, model_name, not_pos = True, lemmad = True, print_vocab = False):
		label_name = file_name_2[:-4] + "_labels.npy"
		#if file_name_2 exists, then all np arrays exists. load them and do BOW
		if os.path.isfile(file_name_2):
			numpy_counts = np.load(file_name_2)
			numpy_cum_labels = np.load(label_name)
			count_vectors = numpy_counts.tolist()
			list_labels = numpy_cum_labels.tolist()
			numpy_cumulative = np.load(file_name_1)
			cumulative_word_vec = numpy_cumulative.tolist()
		else:
			if os.path.isfile(file_name_1) : #check if file_name 1 exists, and load if it does
				numpy_cumulative = np.load(file_name_1)
				cumulative_word_vec = numpy_cumulative.tolist()

			else: #otherwise, load the correct json
				if not_pos:
					articles = self.read_data(path=ApplicationConstants.all_articles_random_v4_cleaned, number_of_articles=1000
											  ,save=False)
				else:
					articles = self.read_data(path = ApplicationConstants.all_articles_random_v4_cleaned_pos_candidate_names,
											  number_of_articles =1000, save = False)

				#create the cumulative word vec for all articles, and save it as numpy array in store directory
				cumulative_word_vec = self.calc_word_vector(articles, not_pos, lemmad, print_vocab)
				numpy_cumulative = np.array(cumulative_word_vec)
				np.save(file_name_1, numpy_cumulative)
				print("store/total num words = " + str(len(cumulative_word_vec)))

				#get all articles in a list
			list_articles_list = []
			list_labels = []
			for i, split in enumerate(articles):
				print("Fold " + str(i + 1))
				for j, leaning in enumerate(split):
					if i is 0:
						training_dataset = split[leaning][ApplicationConstants.Train]
						validation_dataset = split[leaning][ApplicationConstants.Validation]
						test_dataset = split[leaning][ApplicationConstants.Test]
						articles_list = list(map(lambda article: article.Content, training_dataset +
												 validation_dataset + test_dataset))
						list_articles_list.append(articles_list)
						labels = list(map(lambda article: article.Label.TargetGender, training_dataset +
										  validation_dataset +test_dataset))
						list_labels.append(labels)
					else:
						break
				if i > 0:
					break
			articles_list = [j for sub in list_articles_list for j in sub]
			labels = [j for sub in list_labels for j in sub]

			#zip and shuffle the list of articles
			print("zipping and shuffling")
			zippedArticles = list(zip(articles_list, labels))
			random.shuffle(zippedArticles)

			list_articles = []
			list_labels = []
			print("unzipping")
			for article, label in zippedArticles:
				list_articles.append(article)
				list_labels.append(label)

			#change the 0 labels to -1 for easier training
			print("enumerating")
			for i, label in enumerate(list_labels):
				if label == 0:
					list_labels[i] = -1

			#Create a word count vector for every article in the dataset and save the count vector in numpy array
			print("appending")
			count_vectors = []
			i = 0
			nlp = spacy.load("en_core_web_lg")
			for article in list_articles:
				count_vectors.append(self.calc_count_doc_count_vector(cumulative_word_vec, article, nlp, lemmad))
			numpy_count = np.array(count_vectors)
			numpy_label = np.array(list_labels)
			np.save(file_name_2, numpy_count)
			np.save(label_name, numpy_label)

		#Build and train an SVM BOW
		trainLen = int(len(count_vectors) * 0.8)
		acc = 0
		print("building net")
		net = SVM()
		print("training")
		net.Train(count_vectors[:trainLen], list_labels[:trainLen], count_vectors[trainLen:], list_labels[trainLen:])
		weights = net.Get_Weights()
		predictions = net.Predict(count_vectors[trainLen:])

		acc = accuracy_score(list_labels[trainLen:], predictions)
		target_names = ['Female', 'Male']
		print("accuracy is: " + str(acc))

		#if the accuracy is high enough, print the metrics, and print top words to a file
		if acc >= 0.60:
			print(classification_report(list_labels[trainLen:], predictions, target_names=target_names))

			weights = weights[0]

			resTop = sorted(range(len(weights)), key=lambda sub: weights[sub])[-21:]
			resBottom = sorted(range(len(weights)), key=lambda sub: weights[sub])[:21]
			model_name_amp = model_name + "_" + str(acc) + "_.sav"
			pickle.dump(net, open(model_name_amp, 'wb'))
			fout = open('output_words.txt', 'w')
			fout.write("Male Top Words: \n")
			for index in resTop:
				fout.write(cumulative_word_vec[index] + ' ' + str(float(weights[index])) + '\n')
			fout.write("Female Top Words: \n")
			for index in resBottom:
				fout.write(cumulative_word_vec[index] + ' ' + str(float(weights[index])) + '\n')


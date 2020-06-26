from DataReader import DataReader
from DataContracts import Article
from doc2vec import doc
from Metrics import Metrics
from Visualizer import Visualizer

import ApplicationConstants

import re

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from Models.NN_engine import NN

import numpy as np

import os.path



class pretrain():

    def __init__(self):

        self.Visualizer = Visualizer()

    def filter_ATN_content(self, content, publication=None):
        content = re.sub("â€œ|â€\?|“|”|\"\"", '"', content)
        content = re.sub("â€œ|â€\?|” | ”|\"\"", '"', content)
        content = re.sub('â€"', '—', content)
        content = content.lower()
        content = re.sub('(?:https?:)?//[\w\d]+\.[\w\d/\.]+', '', content)
        content = re.sub('[\w\d/\.]+(?:\.com|\.net|\.org|\.co)[\w\d/\.]*', '', content)
        content = re.sub(' {2,}', ' ', content)
        content = re.sub(r"#(\w+)", '', content)
        content = re.sub(r"@(\w+)", '', content)
        content = re.sub("reuters", '', content)
        content = re.sub("(reuters)", '', content)
        content = re.sub("\\/", ' ', content)
        content = re.sub("(?<=/)[^/]+(?=/)", ' ', content)
        content = re.sub("\\n", '', content)
        content = re.sub("\\'s", '\'s', content)
        content = re.sub("\\'t", '\'t', content)
        content = re.sub("\\'d", '\'d', content)
        content = re.sub("\\'re", '\'re', content)
        content = re.sub("\\\'", '\'', content)
        content = re.sub("\\xa0", ' ', content)
        content = re.sub("\(\)", '', content)
        content = re.sub(" ing ", '', content)

        return content
    def pretrain_and_fineTune(self, dirty=True, notBaseline = True):
        reader = DataReader()  # adds 2 G
        #input("Press Enter to continue...")
        portionToLoad = 0.20
        print("Loading %.2f All The News" % portionToLoad)
        if notBaseline:
            if (os.path.exists('store/pretrained_model.model')) == False:
                all_the_news = reader.Load_newer_ATN(ApplicationConstants.all_the_news_newer_path, portionToLoad)

                dirty_pretrain_content = list(map(lambda article: article.Content,
                                                  all_the_news))  # grabbing only half cuz my computer can't fit training all this in memory


                pretrain_content = []
                print("Cleaning All The News")
                for article in dirty_pretrain_content:
                    pretrain_content.append(self.filter_ATN_content(article))


                dirty_pretrain_content.clear()
                del dirty_pretrain_content

                pretrain_labels = list(
                    map(lambda article: article.Label, all_the_news))  # these values are null since ATN doesn't have gender labels
                del all_the_news
                del reader

            print("Opening file to append to")

            pretrain_epochs = [25]#[25, 50, 100]#, 200] #fix the next 5 lines
            fineTune_epochs = [100]#[ 25, 50, 100]#, 200]
            vector_sizes = [100]#[20, 100, 300]
            avFile = "pretrain_and_cleaned_fineTune_av_v4.txt"#"pretrain_and_fineTune_atnClean_av.txt"
            allfile = "pretrain_and_cleaned_fineTune_v4.txt"#"pretrain_and_fineTune_atnClean.txt"

        else:
            pretrain_epochs = [0]
            fineTune_epochs = [100]
            vector_sizes =[50]
            avFile = "pretrain_and_fineTune_nopretrain_av.txt"
            allfile = "pretrain_and_fineTune_nopretrain_oldnums.txt"

        for pretrain_epoch in pretrain_epochs:
            for fineTune_epoch in fineTune_epochs:
                for vector_size in vector_sizes:

                    if (pretrain_epoch == 0 and fineTune_epoch == 0 and vector_size == 100 and notBaseline == True) or (pretrain_epoch == 0 and fineTune_epoch == 0 and vector_size == 300) or (pretrain_epoch == 0 and fineTune_epoch == 0 and vector_size == 20):
                        print("skipping")
                        print(pretrain_epoch, fineTune_epoch, vector_size)
                        continue

                    else:
                        print("doing")
                        print(pretrain_epoch, fineTune_epoch, vector_size)
                        with open(avFile, "a+") as f_av:
                            with open(allfile, "a+") as f:

                                # else:
                                if notBaseline:
                                    print("Pretraining")

                                    docEmbed = doc()
                                    if (os.path.exists('store/pretrained_model.model')):
                                        pretrained_article_model = docEmbed.Load_Model('store/pretrained_model.model')
                                    else:
                                        pretrained_article_model = docEmbed.Embed(pretrain_content, pretrain_labels,
                                                                                       vector_size=vector_size,
                                                                                       epochs=pretrain_epoch,
                                                                                       lower=False)  # started with 2. was not working. 20 worked well
                                        pretrained_article_model.save('store/pretrained_model.model')
                                    del docEmbed
                                reader = DataReader()
                                if dirty:
                                    finetuneSet = reader.Load_Splits(ApplicationConstants.all_articles_random_v4, None,
                                                                     number_of_articles=50,
                                                                     clean=False, save=False, shouldRandomize=False)
                                else:
                                    finetuneSet = reader.Load_Splits(ApplicationConstants.all_articles_random_v4_cleaned, None,
                                                                     number_of_articles=50,
                                                                     clean=False, save=False, shouldRandomize=False)
                                del reader
                                breitbartTtlPrecision, foxTtlPrecision, usaTtlPrecision, huffTtlPrecision, nytTtlPrecision = 0, 0, 0, 0, 0
                                breitbartTtlRecall, foxTtlRecall, usaTtlRecall, huffTtlRecall, nytTtlRecall = 0, 0, 0, 0, 0
                                breitbartTtlF1, foxTtlF1, usaTtlF1, huffTtlF1, nytTtlF1 = 0, 0, 0, 0, 0
                                if notBaseline:
                                    print("Fine tuning folds")

                                #f.write("Training vector size " + str(vector_size) + " pretrain " + str(
                                #    pretrain_epoch) + " finetune " + str(fineTune_epoch) + "\n")
                                #f_av.write("Training vector size " + str(vector_size) + " pretrain " + str(
                                #    pretrain_epoch) + " finetune " + str(fineTune_epoch) + "\n")

                                print("Training vector size " + str(vector_size) + " pretrain " + str(
                                    pretrain_epoch) + " finetune " + str(fineTune_epoch))

                                for i, split in enumerate(finetuneSet):
                                    print("Fold " + str (i+1))
                                    for j, leaning in enumerate(split):
                                        training_dataset = split[leaning][ApplicationConstants.Train]
                                        validation_dataset = split[leaning][ApplicationConstants.Validation]
                                        test_dataset = split[leaning][ApplicationConstants.Test]

                                        fineTune_train_articles = list(
                                            map(lambda article: article.Content, training_dataset + validation_dataset))
                                        fineTune_train_labels = list(
                                            map(lambda article: article.Label.TargetGender, training_dataset + validation_dataset))
                                        fineTune_test_articles = list(map(lambda article: article.Content, test_dataset))
                                        fineTune_test_labels = list(map(lambda article: article.Label.TargetGender, test_dataset))

                                        docEmbed = doc()
                                        if notBaseline:
                                            fine_tuned_model = docEmbed.fine_tune(fineTune_train_articles + fineTune_test_articles, fineTune_train_labels + fineTune_test_labels,
                                                                                       pretrained_article_model, fineTune_epoch)
                                            #fine_tuned_model = docEmbed.fine_tune(fineTune_train_articles ,fineTune_train_labels,
                                            #                                            pretrained_article_model, fineTune_epoch)
                                        else:
                                            fine_tuned_model = docEmbed.Embed(fineTune_train_articles + fineTune_test_articles, fineTune_train_labels + fineTune_test_labels,
                                                                                      vector_size=vector_size,
                                                                                      epochs=fineTune_epoch,
                                                                                      lower=True)
                                        FT_labels, FT_embeddings = docEmbed.gen_vec(fine_tuned_model,
                                                                                                fineTune_train_articles + fineTune_test_articles,
                                                                                                fineTune_train_labels + fineTune_test_labels)
                                        #FT_Train_labels, FT_train_embeddings = docEmbed.gen_vec(fine_tuned_model,
                                        #                                                     fineTune_train_articles,
                                        #                                                      fineTune_train_labels)
                                        #FT_Test_labels, TF_Test_embeddings = docEmbed.gen_vec(fine_tuned_model,
                                        #                                                      fineTune_test_articles,
                                        #                                                    fineTune_test_labels)
                                        FT_labels = list(FT_labels) #train + val
                                        #FT_labels = list(FT_Train_labels)
                                        #FT_test_labels = list(FT_Test_labels) #test labels
                                        del docEmbed

                                        model = NN()
                                        #print(len(FT_embeddings),len(FT_labels), len(training_dataset), len(validation_dataset), len(test_dataset))
                                        model.Train(FT_embeddings[:len(training_dataset) ], FT_labels[:len(training_dataset)],
                                                    FT_embeddings[len(training_dataset):len(training_dataset) + len(validation_dataset)], FT_labels[len(training_dataset):len(training_dataset) + len(validation_dataset)])
                                        prediction = model.Predict(FT_embeddings[len(training_dataset) + len(validation_dataset):])
                                        #model.Train(FT_train_embeddings[:len(training_dataset)], FT_labels[:len(training_dataset)], FT_train_embeddings[len(training_dataset):], FT_labels[len(training_dataset):])
                                        #prediction = model.Predict(TF_Test_embeddings)

                                        Met = Metrics()
                                        if j == 0:
                                            lean = "Breitbart"
                                            print(len(prediction), len(FT_labels), len(training_dataset) + len(validation_dataset))
                                            breitbartTtlPrecision += Met.Precision(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            breitbartTtlRecall += Met.Recall(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            breitbartTtlF1 += Met.Fmeasure(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            #breitbartTtlPrecision += Met.Precision(prediction, FT_test_labels)
                                            #breitbartTtlRecall += Met.Recall(prediction, FT_test_labels)
                                            #breitbartTtlF1 += Met.Fmeasure(prediction, FT_test_labels)
                                        if j == 1:
                                            lean = "Fox"
                                            foxTtlPrecision += Met.Precision(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            foxTtlRecall += Met.Recall(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            foxTtlF1 += Met.Fmeasure(prediction,FT_labels[len(training_dataset) + len(validation_dataset):])
                                            #foxTtlPrecision += Met.Precision(prediction, FT_test_labels)
                                            #foxTtlRecall += Met.Recall(prediction, FT_test_labels)
                                            #foxTtlF1 += Met.Fmeasure(prediction,FT_test_labels)
                                        if j == 2:
                                            lean = "USA"
                                            usaTtlPrecision += Met.Precision(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            usaTtlRecall += Met.Recall(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            usaTtlF1 += Met.Fmeasure(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            #usaTtlPrecision += Met.Precision(prediction, FT_test_labels)
                                            #usaTtlRecall += Met.Recall(prediction, FT_test_labels)
                                            #usaTtlF1 += Met.Fmeasure(prediction, FT_test_labels)
                                        if j == 3:
                                            lean = "Huff"
                                            huffTtlPrecision += Met.Precision(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            huffTtlRecall += Met.Recall(prediction,FT_labels[len(training_dataset) + len(validation_dataset):])
                                            huffTtlF1 += Met.Fmeasure(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            #huffTtlPrecision += Met.Precision(prediction, FT_test_labels)
                                            #huffTtlRecall += Met.Recall(prediction, FT_test_labels)
                                            #huffTtlF1 += Met.Fmeasure(prediction, FT_test_labels)
                                        if j == 4:
                                            lean = "NYT"
                                            nytTtlPrecision += Met.Precision(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            nytTtlRecall += Met.Recall(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            nytTtlF1 += Met.Fmeasure(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])
                                            #nytTtlPrecision += Met.Precision(prediction, FT_test_labels)
                                            #nytTtlRecall += Met.Recall(prediction,  FT_test_labels)
                                            #nytTtlF1 += Met.Fmeasure(prediction,  FT_test_labels)
                                        print("Leaning:", lean, "precision:",
                                              Met.Precision(prediction, FT_labels[len(training_dataset) + len(validation_dataset):]), "recall:",
                                              Met.Recall(prediction, FT_labels[len(training_dataset) + len(validation_dataset):]), "F-Measure:",
                                              Met.Fmeasure(prediction, FT_labels[len(training_dataset) + len(validation_dataset):]))

                                        #print("Leaning:", lean, "precision:",
                                        #      Met.Precision(prediction,FT_test_labels), "recall:",
                                        #      Met.Recall(prediction,FT_test_labels),  "F-Measure:",
                                        #      Met.Fmeasure(prediction,FT_test_labels))


                                        #f.write("Leaning: " + lean + " precision:" +
                                        #       str(Met.Precision(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])) + " recall: " +
                                        #        str(Met.Recall(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])) + " F-Measure: " +
                                        #        str(Met.Fmeasure(prediction, FT_labels[len(training_dataset) + len(validation_dataset):])) + "\n")

                                        #f.write("Leaning:" + lean + "precision:" +
                                        #      str(Met.Precision(prediction, FT_test_labels)) + "recall:"+
                                        #      str(Met.Recall(prediction, FT_test_labels))+ "F-Measure:" +
                                        #      str(Met.Fmeasure(prediction, FT_test_labels)) + "\n")
                                        del model

                                        del Met

                                        if i == 0:
                                            self.Visualizer.plot_TSNE(leaning,
                                                                      FT_embeddings,
                                                                      FT_labels ,
                                                                      training_dataset + validation_dataset + test_dataset)
                                            #self.Visualizer.plot_TSNE(leaning,
                                            #                          FT_train_embeddings + TF_Test_embeddings,
                                            #                          FT_labels + FT_test_labels,
                                            #                          training_dataset + validation_dataset + test_dataset)
                                '''
                                f_av.write("Average Breitbart Recall: " + str(breitbartTtlRecall / 5) + " Average Breitbart Precision: " + str(
                                    breitbartTtlPrecision / 5) + " Average Breitbart F1: " + str(breitbartTtlF1 / 5) + "\n")
                                f_av.write("Average Fox Recall: " + str(
                                    foxTtlRecall / 5) + " Average Fox Precision: " + str(
                                    foxTtlPrecision / 5) + " Average Fox F1: " + str(foxTtlF1 / 5) + "\n")
                                f_av.write("Average USA Recall: " + str(
                                    usaTtlRecall / 5) + " Average USA Precision: " + str(
                                    usaTtlPrecision / 5) + " Average USA F1: " + str(usaTtlF1 / 5) + "\n")
                                f_av.write("Average Huffpost Recall: " + str(
                                    huffTtlRecall / 5) + " Average Huffpost Precision: " + str(
                                    huffTtlPrecision / 5) + " Average Huffpost F1: " + str(huffTtlF1 / 5) + "\n")
                                f_av.write("Average NYT Recall: " + str(
                                    nytTtlRecall / 5) + " Average NYT Precision: " + str(
                                    nytTtlPrecision / 5) + " Average NYT F1: " + str(nytTtlF1 / 5) + "\n")
                                '''
                                print("Average Breitbart Recall: " + str(
                                    breitbartTtlRecall / 5) + " Average Breitbart Precision: " + str(
                                    breitbartTtlPrecision / 5) + " Average Breitbart F1: " + str(breitbartTtlF1 / 5))
                                print("Average Fox Recall: " + str(
                                    foxTtlRecall / 5) + " Average Fox Precision: " + str(
                                    foxTtlPrecision / 5) + " Average Fox F1: " + str(foxTtlF1 / 5))
                                print("Average USA Recall: " + str(
                                    usaTtlRecall / 5) + " Average USA Precision: " + str(
                                    usaTtlPrecision / 5) + " Average USA F1: " + str(usaTtlF1 / 5))
                                print("Average Huffpost Recall: " + str(
                                    huffTtlRecall / 5) + " Average Huffpost Precision: " + str(
                                    huffTtlPrecision / 5) + " Average Huffpost F1: " + str(huffTtlF1 / 5))
                                print("Average NYT Recall: " + str(
                                    nytTtlRecall / 5) + " Average NYT Precision: " + str(
                                    nytTtlPrecision / 5) + " Average NYT F1: " + str(nytTtlF1 / 5))







    # article_labels, article_embeddings = self.docEmbed.gen_vec(pretrained_article_model, articles, labels)
    #
    def print_all_the_news(self):
        reader = DataReader()
        all_the_news = reader.Load_ATN(ApplicationConstants.all_the_news_path)
        pretrain_content = list(map(lambda article: article.Content, all_the_news))[:int(
            len(all_the_news) * 0.25)]  # grabbing only half cuz my computer can't fit training all this in memory

        for article in pretrain_content:
            print(article)
            input("Press Enter to continue...")



pf = pretrain()
pf.pretrain_and_fineTune(dirty = False, notBaseline=True) #run pretrain and fineTune on atn, then on cleaned newsbias dataset

#from Orchestrator import Orchestrator
#orchestrator = Orchestrator()

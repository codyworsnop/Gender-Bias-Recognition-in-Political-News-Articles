from DataReader import DataReader
from DataContracts import Article
from doc2vec import doc
from Metrics import Metrics

import ApplicationConstants

import re

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from Models.NN_engine import NN

import numpy as np

import os.path

def filter_ATN_content( content, publication=None):
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


def pretrain_and_fineTune( dirty=True):
    reader = DataReader()  # adds 2 G
    input("Press Enter to continue...")
    portionToLoad = 0.001
    print("Loading %.2f All The News" % portionToLoad)
    all_the_news = reader.Load_newer_ATN(ApplicationConstants.all_the_news_newer_path, portionToLoad)

    dirty_pretrain_content = list(map(lambda article: article.Content,
                                      all_the_news))  # grabbing only half cuz my computer can't fit training all this in memory
    input("Press Enter to continue...")
    # print(dirty_pretrain_content[:10])
    # print()

    pretrain_content = []
    print("Cleaning All The News")
    for article in dirty_pretrain_content:
        pretrain_content.append(filter_ATN_content(article))

    input("Press Enter to continue...")
    dirty_pretrain_content.clear()
    del dirty_pretrain_content
    input("Press Enter to continue...")
    pretrain_labels = list(
        map(lambda article: article.Label, all_the_news))  # these values are null since ATN doesn't have gender labels
    del all_the_news
    del reader
    # print(len(pretrain_content), pretrain_labels[:10])
    # nums = random.sample(range(0, 20000), 10)
    # for num in nums:
    # print(pretrain_content[num])
    # jkhkjhkjhk
    print("Opening file to append to")
    f = open("pretrain_and_fineTune_cleaned.txt", "a+")
    pretrain_epochs = [10, 25, 50, 100, 200, 500]
    fineTune_epochs = [10, 25, 50, 100, 200, 500]
    vector_sizes = [20, 100, 200, 300, 500, 1000]

    for vector_size in vector_sizes:
        for pretrain_epoch in pretrain_epochs:
            for fineTune_epoch in fineTune_epochs:
                # if (os.path.exists('store/pretrained_model.model')):
                #	pretrained_article_model = self.docEmbed.Load_Model('store/pretrained_model.model')
                # else:
                print("Pretraining")
                docEmbed = doc()
                pretrained_article_model = docEmbed.Embed(pretrain_content, pretrain_labels,
                                                               vector_size=vector_size,
                                                               epochs=pretrain_epoch,
                                                               lower=False)  # started with 2. was not working. 20 worked well
                # pretrained_article_model.save('store/pretrained_model.model')
                del docEmbed
                reader = DataReader()
                if dirty:
                    finetuneSet = reader.Load_Splits(ApplicationConstants.all_articles_random_v2, None,
                                                     number_of_articles=50,
                                                     clean=False, save=False, shouldRandomize=False)
                else:
                    finetuneSet = reader.Load_Splits(ApplicationConstants.all_articles_random_v2_cleaned, None,
                                                     number_of_articles=50,
                                                     clean=False, save=False, shouldRandomize=False)
                del reader
                precisionTtl = 0
                recallTtl = 0
                f1Ttl = 0
                print("Fine tuning folds")
                for i, leaning in enumerate(finetuneSet[0]):
                    training_dataset = finetuneSet[0][leaning][ApplicationConstants.Train]
                    validation_dataset = finetuneSet[0][leaning][ApplicationConstants.Validation]
                    test_dataset = finetuneSet[0][leaning][ApplicationConstants.Test]

                    fineTune_train_articles = list(
                        map(lambda article: article.Content, training_dataset + validation_dataset))
                    fineTune_train_labels = list(
                        map(lambda article: article.Label.TargetGender, training_dataset + validation_dataset))
                    fineTune_test_articles = list(map(lambda article: article.Content, test_dataset))
                    fineTune_test_labels = list(map(lambda article: article.Label.TargetGender, test_dataset))
                    docEmbed = doc()
                    fine_tuned_model = docEmbed.fine_tune(fineTune_train_articles, fineTune_train_labels,
                                                               pretrained_article_model, fineTune_epoch)
                    FT_Train_labels, FT_Train_embeddings = docEmbed.gen_vec(fine_tuned_model,
                                                                                 fineTune_train_articles,
                                                                                 fineTune_train_labels)
                    FT_Test_labels, TF_Test_embeddings = docEmbed.gen_vec(fine_tuned_model, fineTune_test_articles,
                                                                               fineTune_test_labels)
                    print("Training vector size " + str(vector_size) + " pretrain " + str(
                        pretrain_epoch) + " finetune " + str(fineTune_epoch))
                    del docEmbed
                    model = NN()
                    model.Train(FT_Train_embeddings[:len(training_dataset)], FT_Train_labels[:len(training_dataset)],
                                FT_Train_embeddings[len(training_dataset):], FT_Train_labels[len(training_dataset):])
                    prediction = model.Predict(TF_Test_embeddings)
                    Met = Metrics()
                    print("Model:", str(type(model)).split('.')[2].split('\'')[0], "precision:",
                          Met.Precision(prediction, FT_Test_labels), "recall:",
                          Met.Recall(prediction, FT_Test_labels), "F-Measure:",
                          Met.Fmeasure(prediction, FT_Test_labels))
                    if i == 0:
                        f.write("Training vector size " + str(vector_size) + " pretrain " + str(
                            pretrain_epoch) + " finetune " + str(fineTune_epoch) + "\n")
                    f.write("Model: " + str(type(model)).split('.')[2].split('\'')[0] + " precision:" +
                            str(Met.Precision(prediction, FT_Test_labels)) + " recall: " +
                            str(Met.Recall(prediction, FT_Test_labels)) + " F-Measure: " +
                            str(Met.Fmeasure(prediction, FT_Test_labels)) + "\n")
                    del model
                    recallTtl += Met.Recall(prediction, FT_Test_labels)
                    precisionTtl += Met.Precision(prediction, FT_Test_labels)
                    f1Ttl += Met.Fmeasure(prediction, FT_Test_labels)
                    del Met
                    if i == 4:
                        f.write("Average recall: " + str(recallTtl / 5) + " Average precision: " + str(
                            precisionTtl / 5) + " Average F1 " + str(f1Ttl / 5) + "\n")
    f.close()


# article_labels, article_embeddings = self.docEmbed.gen_vec(pretrained_article_model, articles, labels)
#
def print_all_the_news():
    reader = DataReader()
    all_the_news = reader.Load_ATN(ApplicationConstants.all_the_news_path)
    pretrain_content = list(map(lambda article: article.Content, all_the_news))[:int(
        len(all_the_news) * 0.25)]  # grabbing only half cuz my computer can't fit training all this in memory

    for article in pretrain_content:
        print(article)
        input("Press Enter to continue...")

pretrain_and_fineTune(True)
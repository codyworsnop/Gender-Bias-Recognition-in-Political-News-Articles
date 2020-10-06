#################################
# run_BOW.py:
# This file will generate the different BOW findings.
#################################

from Orchestrator import Orchestrator
import ApplicationConstants
import spacy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
from Models.SVM_engine import SVM

orchestrator = Orchestrator()

'''
Uncomment the option you wish to run; our "overall words" were obtained with  option 1, our "adjective words" were 
obtained with option 2. You can turn print_vocab off without affecting results. 
file_name_1 is the cumulative word vector for all words in the articles
file_name_2 is a vector of vectors for the words in each article
model_name is the name of the svm trained to create the BOW
not_pos is a bool that determines if the BOW is run on all words or adj. If True, all words
lemmad is a bool that determines if a lemma is applied to the words. if True, a lemma is applied
print_vocab is a bool that determines if all of the vocab from the BOW is printed- helps confirm proper cleaning occurs

NOTE: to run this file, articles must have been collected and run_preprocessor.py must have been run
'''

#OPTION 1: run run_bow on all words in vocab, without lemma, print vocab to confirm data is cleaned properly
#orchestrator.run_bow( "store/np_cumulative_vec_ALLnoL_HC.npy", "store/np_count_vec_ALLnoL_HC.npy", "store/perceptron_ALLnoL_HC.sav", "Hillary_Clinton", True, False, True, True, False) #notPos, lemmad, printvocab

#OPTION 2: run run_bow on adjectives in vocab, without lemma, print vocab to confirm proper cleaning
#orchestrator.run_bow( "store/np_cumulative_vec_ADJnoL_4_all.npy", "store/np_count_vec_ADJnoL_4_all.npy", "store/perceptron_ADJnoL_4_all.sav", HillaryClinton, False, False, True, False) #Pos, lemmad, printvocab


#OPTION 3: run run_bow on all words in vocab, with lemma, print vocab to confirm data is cleaned properly
#orchestrator.run_bow( "store/np_cumulative_vec_ALLl.npy", "store/np_count_vec_ALLl.npy", "store/perceptron_ALLl.sav",True, True, True) #notPos, lemmad, printvocab

#OPTION 4: run run_bow on adjectives in vocab, with lemma, print vocab to confirm proper cleaning
#orchestrator.run_bow( "store/np_cumulative_vec_ADJl.npy", "store/np_count_vec_ADJl.npy", "store/perceptron_ADJl.sav",False, True, True) #Pos, lemmad, printvocab

#OPTION 5: run run_bow on all words in vocab without lemma, print vocab to confirm that data is cleaned properly, run it on data in folds
#orchestrator.run_bow("store/np_cumulative_vec_ALLnoLFolds.")



def one_person_holdOut():
    person = "Sarah_Palin"
    articles = orchestrator.read_data(path=ApplicationConstants.all_articles_random_v4_cleaned_pos_candidate_names,
        number_of_articles=50, save=False)
    cumulative_word_vec = orchestrator.calc_word_vector(articles, True, False, True)


    articles_list = list(map(lambda leaning: articles[4][leaning][ApplicationConstants.Train] +
                                             articles[4][leaning][ApplicationConstants.Validation] +
                                             articles[4][leaning][ApplicationConstants.Test], articles[4]))
    articles_list = [item for sublist in articles_list for item in sublist]
    train_articles = list(filter(lambda article: article.Label.TargetName != person,
                                 articles_list))  # change this line
    test_articles = list(filter(lambda article: article.Label.TargetName == person,
                                articles_list))  # change this line
    train_articles = train_articles + test_articles
    train_content = list(map(lambda article: article.Content, train_articles))
    train_label = list(map(lambda article: article.Label.TargetGender, train_articles))



    #test_content = list(map(lambda article: article.Content, test_articles))
    #test_label = list(map(lambda article: article.Label.TargetGender, test_articles))

    #labels = train_label + test_label
    labels = train_label
    for i, label in enumerate(labels):  # was list_labels
        if label == 0:
            labels[i] = -1  # was list_labels
    count_vectors = []
    i = 0
    nlp = spacy.load("en_core_web_lg")
    for article in train_content:
        count_vectors.append(orchestrator.calc_count_doc_count_vector(cumulative_word_vec, article, nlp, False))
    #for article in test_content:
    #    count_vectors.append(orchestrator.calc_count_doc_count_vector(cumulative_word_vec, article, nlp, False))

    trainLen = len(train_articles)
    diffLen = len(train_articles) - len(test_articles)

    acc = 0
    print("building net")
    net = SVM()
    print("training")
    # print(len(labels), len(labels[:trainLen]))
    print(trainLen, diffLen)
    print(len(count_vectors), len(count_vectors[:diffLen]))


    net.Train(count_vectors[:diffLen], labels[:diffLen], count_vectors[:diffLen],
              labels[:diffLen])  # was list_labels
    weights = net.Get_Weights()
    #predictions = net.Predict(count_vectors[trainLen:])
    predictions = net.Predict(count_vectors[diffLen:])
    # print("trainLen", str(trainLen))
    print(len(predictions), len(labels[diffLen:]))

    #acc = accuracy_score(labels[trainLen:], predictions)  # was list_labels
    acc = accuracy_score(labels[diffLen:], predictions)
    #tp = 0
    #testLabels = labels[diffLen:]
    #for i, prediction in enumerate(predictions):
    #    if prediction == testLabels[i]:
    #        tp+=1
    #acc = tp/len(testLabels)




    target_names = ['Female', 'Male']
    print("accuracy is: " + str(acc))

    # if the accuracy is high enough, print the metrics, and print top words to a file

    #print(classification_report(labels[trainLen:], predictions, target_names=target_names))  # was list_labels
    print(classification_report(labels[diffLen:], predictions, target_names=target_names))
    weights = weights[0]

    resTop = sorted(range(len(weights)), key=lambda sub: weights[sub])[-25:]
    resTop.reverse() #put from largest to smallest
    resBottom = sorted(range(len(weights)), key=lambda sub: weights[sub])[:25]
    model_name_amp ="store/perceptron_ALLnoL_HC.sav"
    pickle.dump(net, open(model_name_amp, 'wb'))
    fout = open('output_words' + person + '.txt', 'w')
    fscores = open('output_word_scores_by_person.txt', 'a')
    fscores.write("\n\n"+person + "\n")
    fscores.write("Accuracy is: " + str(acc) +"\n")
    fscores.write(str(classification_report(labels[diffLen:], predictions, target_names=target_names)) + "\n")
    fout.write("Male Top Words: \n")
    for index in resTop:
        fout.write(cumulative_word_vec[index] + ' ' + str(float(weights[index])) + '\n')
    fout.write("Female Top Words: \n")
    for index in resBottom:
        fout.write(cumulative_word_vec[index] + ' ' + str(float(weights[index])) + '\n')
    fout.close()
    fscores.close()

one_person_holdOut()
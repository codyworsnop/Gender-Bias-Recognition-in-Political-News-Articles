#######This file calculates accuracy, fmeasre, recall, and precision #######
from sklearn.metrics import accuracy_score

class Metrics():

    def Accuracy(self, prediction, labels): 
       return accuracy_score(labels, prediction)
        
    def Fmeasure(self, prediction, labels):

        precision = self.Precision(prediction, labels)
        recall = self.Recall(prediction, labels) 

        if precision > 0 and recall > 0:

            mult = 2 * precision * recall
            addit = precision + recall

            return mult/addit
        else:
            return 0

    def Recall(self, prediction, labels):
        tp, _, _, fn = self.__Condition(prediction, labels)
        
        if tp > 0:
            return (tp/(tp+fn))
        else:
            return 0

    def Precision(self, prediction, labels):
        tp, _, fp, _ = self.__Condition(prediction, labels)

        if tp > 0:
            return (tp/(tp+fp))
        else:
            return 0

    def __Condition(self, prediction, truth_labels):

        tp, tn, fp, fn = 0, 0, 0, 0

        for i in range(len(prediction)):

            if prediction[i] == 0 and truth_labels[i] == 0: 
                tp+=1
            if prediction[i] == 1 and truth_labels[i] == 1:
                tn +=1
            if prediction[i] == 0 and truth_labels[i] != 0:
                fp +=1
            if prediction[i] == 1 and truth_labels[i] != 1:
                fn +=1
        return tp, tn, fp, fn
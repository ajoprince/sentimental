#Importing our models
from models_b import NN, XGB
from models_sk import MNB, BNB, LSVC, NSVC, LR
from data_processor import feature_find, feature_find_and_index

#Creating Vote CLassifier CLass - only vote if three classifiers agree
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    def classify(self, feature, feature_b):
        votes = []
        try:
            for c in self._classifiers: #inlude XGB, NN way of classifier
                vote = c.classify(feature)
                votes.append(vote)
        except AttributeError:
            vote = c.predict(feature_b).item()
            votes.append(vote)            
        return mode(votes)
    def confidence(self, feature, feature_b):
        votes = []
        try:
            for c in self._classifiers: #inlude XGB, NN way of classifier
                vote = c.classify(feature)
                votes.append(vote)
        except AttributeError:
            vote = c.predict(feature_b).item()
            votes.append(vote)
        s = votes.count(mode(votes))
        return (s/len(votes))*100
    
votedclassifier = VoteClassifier(MNB, BNB, LSVC, NSVC, LR, 
                                  XGB, NN
                                 ) 

       
def sentiment(line):
    feature, feature_b = feature_find(line), feature_find_and_index(line)
    print('Prediction: {} with confidence: {}%' .format(votedclassifier.classify(feature, feature_b), votedclassifier.confidence(feature, feature_b)))  
    


        

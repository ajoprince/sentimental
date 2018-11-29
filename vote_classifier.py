import pickle
from data_processor import feature_find
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
from statistics import mode


#Load our featureset created in data_processor
load = open('featureset.pickle', 'rb')
featureset = pickle.load(load)
load.close()    

#Load our models
load = open('pickled_models\MNB.pickle', 'rb')
MNB = pickle.load(load)
load.close()

load = open('pickled_models\BNB.pickle', 'rb')
BNB = pickle.load(load)
load.close()

load = open('pickled_models\LSVC.pickle', 'rb')
LSVC = pickle.load(load)
load.close()

load = open('pickled_models\SVC.pickle', 'rb')
NSVC= pickle.load(load)
load.close()

load = open('pickled_models\LR.pickle', 'rb')
LR = pickle.load(load)
load.close()    

#Creating Vote CLassifier CLass - only vote if three classifiers agree
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    def classify(self, feature):
        votes = []
        for c in self._classifiers:  
            vote = c.classify(feature)
            votes.append(vote)
        return mode(votes)
    def confidence(self, feature):
        votes = []
        for c in self._classifiers:
            vote = c.classify(feature)
            votes.append(vote)
        s = votes.count(mode(votes))
        return (s/len(votes))*100
    
    
votedclassifier = VoteClassifier(MNB, BNB, LSVC, NSVC, LR) 

       
def sentiment(line):
    feature = feature_find(line)
    print('Prediction: {} with confidence: {}%' .format(votedclassifier.classify(feature), votedclassifier.confidence(feature)))  
    
        

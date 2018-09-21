
#Import dataset from base
from data_processor import dataset #importing dataset from base or load pickled dataset 
import nltk
import pickle

#Shuffle data and split into training and validation sets
import random
random.Random(0).shuffle(dataset) # shuffling feature sets using Random to regenerate the same training set
training_set = dataset[:-40]
test_set = dataset[-40:]
X_train = [(line) for (line, cat) in training_set]
y_train = [(cat) for (line, cat) in training_set]

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression

#Creating models
MNB = SklearnClassifier(MultinomialNB())
BNB = SklearnClassifier(BernoulliNB())
LSVC = SklearnClassifier(LinearSVC())
NSVC = SklearnClassifier(NuSVC())
LR = SklearnClassifier(LogisticRegression())

Models = {MNB: 'MNB', BNB: 'BNB', LSVC: 'LSVC', NSVC: 'NSVC', LR: 'LR'}


#Creating loop for model training and pickling
for _ in Models:
    _.train(training_set)
    print('{} Accuracy: {}'.format(Models[_], nltk.classify.accuracy(_,test_set)))
    
'''Pickle Models
for y in Models:
    pop = open('pickled_models/{}.pickle'.format(Models[y]), 'wb') #pickle models to folder named pickled_models 
    pickle.dump(y, pop)
    pop.close()
'''

#Import initial libraries
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

#Importing positive and negative datasets. Each line corresponds to one sample.
neg_data = open('training_neg.csv', 'r').read()
pos_data = open('training_pos.csv', 'r').read()

#We will merge the two datasets into one dataset
data = [] #dataset
all_words = [] #all words in dataset
s=0
n=10000 # number of examples from each set (any limit or none can be chosen)


for line in neg_data.split('\n'):
    if s < n:
        words = word_tokenize(line)
        all_words.extend(words)
        data.append((line, 0)) # creating target vector where 0 is negative and 1 is positive
        s +=1
s=0        
for line in pos_data.split('\n'):
    if s < n:
        words = word_tokenize(line)
        all_words.extend(words)
        data.append((line, 1))
        s +=1
        
#Creating fetureset of the 5000 most informative words 
stop_words = set(stopwords.words('english'))
words = set(nltk.corpus.words.words()) #english words
true_words = []
for word in all_words:
    if word not in stop_words and word in words:  #removing stopwords and non english words
        true_words.append(word.lower())


true_words = nltk.FreqDist(true_words) #?
featureset = list(true_words.keys())[:5000] #5000 most frequent words

'''Create function that finds features in each line of our document 
where the ith column corresponds to the the ith word in featuresets
This will be used to create training data for our XGBoost model and Neural Network
'''
def feature_find_and_index(line):
    words = word_tokenize(line)
    line_vector = np.zeros(len(featureset))
    for feature in featureset:
        line_vector[featureset.index(feature)] = int(feature in words)
    return line_vector    

#Find features of each line in our data document
data_b = [(feature_find_and_index(line), cat) for (line, cat) in data ]

#Advised to pickle
data_mat_p = open('data_b.pickle', 'wb')
pickle.dump(data_b, data_mat_p)
data_mat_p.close()

#Data processing for other models is more straightforward 
'''We now create function similar to above,
but will be used with sklearn models with no need to convert dataset to binary
'''
def feature_find(line): #we can use a similar fuction to find features
    features_vec = {}
    words = word_tokenize(line)
    for w in featureset:
        features_vec[w] = (w in words)
    return features_vec
       

dataset = [(feature_find(line), cat) for (line, cat) in data]

#'Again advised to pickle
dataset_p = open('dataset.pickle', 'wb')
pickle.dump(dataset, data_mat_p)
dataset_p.close()

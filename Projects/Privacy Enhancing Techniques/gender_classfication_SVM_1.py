import re
import os
import pickle
import numpy as np
from nltk.corpus import wordnet
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from nltk.stem import PorterStemmer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

class CustomHTMLParser(HTMLParser):  
    a = ''
    def handle_data(self, data):
        self.a = self.a + str(data)        
    def get_raw_text(self):
        self.a = re.sub(r'[0-9_]+', ' ', self.a)
        self.a = re.sub(r'[^\w\s]', ' ', self.a)        
        return self.a

def preprocess_words(word_list):
    stemmer = PorterStemmer()
    processed_word_list = []
    for word in word_list.split():
        if wordnet.synsets(word):
            processed_word_list.append(stemmer.stem(word))            
    return processed_word_list

def preprocess_data(path = './en/'): 
    all_texts = []
    y_train = []
    for filename in os.listdir(path):    
        root = ET.parse(path + filename).getroot()
        y = 0
        if(root.attrib['gender'] == 'male'):
            y = 1
        for text in root.findall('conversations/conversation'):            
            parser = CustomHTMLParser()
            parser.feed(str(text.text))
            removed_tags = parser.get_raw_text()
            word_list = preprocess_words(removed_tags)    
            all_texts.append(str(word_list))            
            y_train.append(y)    
    return all_texts, y_train

def save_preprocessed_data():     
    with open("x_data_all_svm.txt", "wb") as f:   
        pickle.dump(x_data, f)
    with open("y_data_all_svm.txt", "wb") as f:   
        pickle.dump(y_data, f)

def load_all_data(fx, fy, shuffle=False, seed=1000):
    with open(fx, "rb") as f:   
        x = pickle.load(f)
    with open(fy, "rb") as f:   
        y = pickle.load(f)    
    if shuffle:
        np.random.seed(seed)
        r = np.arange(len(x))
        np.random.shuffle(r)
        x = np.asarray(x)
        y = np.asarray(y)
        x = x[r]
        y = y[r]
    return x,y

def classifier(x_train, x_test, y_train, y_test, iter_list, c_list, kernel = 'rbf'):
    for iteration in iter_list:
        for c in c_list:
            classifierModel = svm.SVC(C = c, kernel = kernel, gamma = 'scale', max_iter = iteration)
            classifierModel.fit(x_train, y_train)
            pred = classifierModel.predict(x_test)
            acc = accuracy_score(pred, y_test)
            print("Iteration : " + str(iteration) + " - C : " + str(c) + " - Accuracy : " + str(acc))

x_data, y_data = preprocess_data()   
save_preprocessed_data()

x_data, y_data = load_all_data("x_data_all_svm.txt", "y_data_all_svm.txt")
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, test_size=0.08)

vectorizer = TfidfVectorizer(max_features=10000)
vectorizer.fit(x_train)
x_train = vectorizer.transform(x_train)
vectorizer.fit(x_test)
x_test = vectorizer.transform(x_test)

c_list = [0.001, 0.1, 1, 10, 100, 1000]

iter_list = [5, 25, 50, 100, 150, 200, 250, 300]
classifier(x_train, x_test, y_train, y_test, iter_list, c_list, 'rbf')
print("******************************************************************")
classifier(x_train, x_test, y_train, y_test, iter_list, c_list, 'linear')
print("******************************************************************")
iter_list = [500, 800, 1000, 1500, 2000]
classifier(x_train, x_test, y_train, y_test, iter_list, c_list, 'rbf')
print("******************************************************************")
classifier(x_train, x_test, y_train, y_test, iter_list, c_list, 'linear')
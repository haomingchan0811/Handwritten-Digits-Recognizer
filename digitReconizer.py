# -*- coding: utf-8 -*-
#!/usr/bin/python

import time, csv, sys
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import naive_bayes
import numpy as np

def load_Training_Data():
    print 'loading Training Data...'
    trainingData = []
    infile = open('train.csv', 'r')
    count = 0
    lines = csv.reader(infile)
    for item in lines:
        if count == 0:
            count += 1
            continue
        label = int(item[0])
        data = item[1:]
        pixel = toMatrix(data)                             # transform into the format of numpy array
        trainingData.append((label, pixel))
    return Combine(trainingData)
    
def load_Testing_Data():
    print 'loading Testing Data...'
    testingData = []
    infile = open('test.csv', 'r')
    count = 0
    lines = csv.reader(infile)
    for item in lines:
        if count == 0:
            count += 1
            continue
        label = 999                                              # default label of the testing set
        data = item[:]
        pixel = toMatrix(data)                             # transform into the format of numpy array
        testingData.append((label, pixel))
    Label, Pixel = Combine(testingData)
    return Pixel

def load_Benchmark_TestResult():
    print 'loading Benchmark Result...'
    label = []
    infile = open('rf_benchmark.csv', 'r')
    lines=csv.reader(infile)
    count = 0
    for line in lines:
        if count == 0:
            count += 1
            continue
        label.append(int(line[1]))     
    label = np.asarray(label)
    return label

def Combine(Data):  
    num = len(Data) 
    Label = np.zeros((num))                                     # Label matrix of all samples
    Pixel = np.zeros((num, 784))                              # Pixel matrix of all samples
    index = 0
    for i in Data:
        Label[index] = i[0]
        Pixel[index] = i[1]
        index += 1
    return (Label, Pixel)

def toMatrix(data):
    pixel = np.zeros((len(data)))
    index = 0
    for i in data:
        i = int(i)
        if i > 60:                              # binarize the matrix
            pixel[index] = 1
        else:
            pixel[index] = 0
        index += 1
    return pixel

def save_Result(prediction):
    print 'saving Prediction result...'                         
    infile = open('svc_Prediction.csv', 'w')
    lines = csv.writer(infile)
    lines.writerow(['ImageId', 'Label'])
    index = 1
    for item in prediction:
        tmp = [index, item]
        lines.writerow(tmp)
        index += 1
        
def svmClassify(trainPixel, trainLabel, testPixel, c, k):
    print 'Training SVM Classifier...'
    # C = 7, kernel = rbf, error = 1629
    classifier =  svm.SVC(C = c, kernel = 'rbf', cache_size= 5000)               #default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’  
    classifier.fit(trainPixel, np.ravel(trainLabel))
    testLabel = classifier.predict(testPixel)
    save_Result(testLabel)
    return testLabel

def random_Forest_Classify(trainPixel, trainLabel, testPixel, n):
    print 'Training Random Forest Classifier...'
    clf = ensemble.RandomForestClassifier(n_estimators=n, max_depth=None, min_samples_split=1, random_state=0) 
    clf.fit(trainPixel, np.ravel(trainLabel))
    testLabel = clf.predict(testPixel)
    save_Result(testLabel)
    return testLabel
    
def knn_Classify(trainPixel, trainLabel, testPixel, n): 
    print 'Training K Neighbors Classifier...'
    clf = neighbors.KNeighborsClassifier(n_neighbors= n)
    clf.fit(trainPixel, np.ravel(trainLabel))
    testLabel = clf.predict(testPixel)
    save_Result(testLabel)
    return testLabel 
       
def gaussian_Naive_Bayes_Classifiy(trainPixel, trainLabel, testPixel):
    print 'Training Gaussian Naive Bayes Classifier...'
    clf = naive_bayes.GaussianNB()
    clf.fit(trainPixel, np.ravel(trainLabel))
    testLabel = clf.predict(testPixel)
    save_Result(testLabel)
    return testLabel
    
def multinomial_Naive_Bayes_Classifiy(trainPixel, trainLabel, testPixel):
    print 'Training Gaussian Naive Bayes Classifier...'
    clf = naive_bayes.MultinomialNB()
    clf.fit(trainPixel, np.ravel(trainLabel))
    testLabel = clf.predict(testPixel)
    save_Result(testLabel)
    return testLabel   

def digitReconition():
    trainLabel, trainPixel = load_Training_Data()
    testPixel = load_Testing_Data()
    for i in xrange(10):
        result = svmClassify(trainPixel, trainLabel, testPixel, i + 1, '')
#         result = random_Forest_Classify(trainPixel, trainLabel, testPixel, i + 1)
#         result = knn_Classify(trainPixel, trainLabel, testPixel, i + 1)
#     result = gaussian_Naive_Bayes_Classifiy(trainPixel, trainLabel, testPixel)
#     result = multinomial_Naive_Bayes_Classifiy(trainPixel, trainLabel, testPixel)
        benchmark = load_Benchmark_TestResult()
        predictionError = 0
        num = testPixel.shape[0]
        for i in xrange(num):
            if result[i] != benchmark[i]:
                predictionError += 1
        print 'Prediction Error is: ', predictionError
        accuracy = (num - predictionError) * 100.0 / num
        print 'Accuracy: %f%%' % accuracy
    
if __name__ == '__main__':
    start = time.clock()
    print '-----------------------------------'
    digitReconition()
    end = time.clock()
    print 'Time: %s min' % ((end - start) / 60)

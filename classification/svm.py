
# SVM implementation
import sys
import util
import time
import math
import random 
import pandas as pd
import MulticlassSVM

import numpy as np


PRINT = True

class SVMClassifier:
  """
  SVM classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.cls = MulticlassSVM.MulticlassSVM(C=0.1, tolorance=0.01, max_iteration=100, random_state=0, verbose=0)
    self.legalLabels = legalLabels
    self.type = "svm"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels);
    self.weights = weights;
  def getNPArray(self, list_dict):
    if isinstance(list_dict[0], dict):
        m = len(list_dict)
        n = len(list_dict[0])
        array = np.asarray(list_dict[0].values())
        for b in range(1, m):
            a = np.asarray(list_dict[b].values())
            array = np.append(array, a, axis=0)
        array = array.reshape(m, n)
    else:
        array = np.asarray(list_dict)
    return array

  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    1. trans trainingData to np.array
    2. trans trainingLabels to np.array
    3. 
    """
    numFeatures = len(trainingData[0])
    numSamples = len(trainingData)
    
    self.features = trainingData[0].keys() # could be useful later
    """
    print type(trainingData)
    print type(trainingData[0])
    print len(trainingData),
    print len(trainingData[0])
    """
    X = self.getNPArray(trainingData)
    #print X
    y = self.getNPArray(trainingLabels)
    #print y
   
    #print trainingData[0].values()
    #array = np.asarray(trainingData[0].values())
    X = X.reshape(numSamples, numFeatures)
    #print X
    #print type(X) >>> np.ndarray
    
    # convert trainingLabels to array
    y = np.asarray(trainingLabels)
    #print type(y) >>> np.array
    self.cls.fit(X, y)
    """
    print self.cls.score(X, y)
    print '======================'
    print 'self.cls.predict()'
    print self.cls.predict(X)
    print y
    print '======================'
    """
    validSamples = self.getNPArray(validationData)
    #print validSamples
    validate_results = self.cls.predict(validSamples)
    validate_y = self.getNPArray(validationLabels)
    count = 0
    for i in range(len(validationLabels)):
        #print validate_y[i], validate_results[i]
        if validate_y[i] == validate_results[i]:
            count += 1

    #print count
  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    testArray = self.getNPArray(data)
    prediction = self.cls.predict(testArray)
    #print prediction
    #print self.cls.W

    #print self.cls.W
    #print len(self.cls.alpha), len(self.cls.alpha[0])
    return prediction


  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    
    featuresWeights = []
    for feature in self.features:
        tpl = (self.weights[label][feature],feature)
        featuresWeights.append(tpl)
    featuresWeights.sort()
    #print featuresWeights
    return [feature for value, feature in featuresWeights[-100:]]
    "*** YOUR CODE HERE ***"
    



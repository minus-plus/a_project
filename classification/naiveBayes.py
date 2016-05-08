# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import sys

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.

  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """

    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]))
    # datum here is a instance of Counter, features here is the coordinates of images

    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]

    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter
    that gives the best accuracy on the held-out validationData.

    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.

    To get the list of all possible features or labels, use self.features and
    self.legalLabels.
    """


    # get the prior frequency of labels
    # print 'length of trainingData is %s' % len(trainingData)
    priorFrequency = util.Counter()
    priorFrequency.incrementAll(trainingLabels, 1)
    # get prior
    self.prior = priorFrequency.copy()
    self.prior.normalize()


    # get the likelihood frequency of features given labels
    likelihoodFrequency = util.Counter()
    for i in range(len(trainingData)):
        label = trainingLabels[i]
        datum = trainingData[i]
        for feature in self.features:
            likelihoodFrequency[(feature, label)] += datum[feature]

    # automatic tuning
    accuracy = -1
    for k in kgrid:
        likelihood = util.Counter()
        # smoothing
        for feature, label in likelihoodFrequency:
            likelihood[(feature, label)] = (likelihoodFrequency[(feature, label)] + k) * 1.0 / (priorFrequency[label] + 2.0 * k)
        self.likelihood = likelihood

        predictions = self.classify(validationData)
        currentAccuracy =  [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)

        if currentAccuracy > accuracy:
            params = (likelihood, k)
            accuracy = currentAccuracy
    self.likelihood, self.k = params
    #print "optimized k is: .2%d" % (self.k)

  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.

    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      #print datum
      #sys.exit(1)
      #datum is Counter type
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses

  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

    To get the list of all possible features or labels, use self.features and
    self.legalLabels.
    """
    logJoint = util.Counter()

    for label in self.legalLabels:
        logJoint[label] = math.log(self.prior[label])
        for feature in datum:
            if datum[feature] > 0:
                logJoint[label] += math.log(self.likelihood[(feature, label)])
            else:
                logJoint[label] += math.log(1 - self.likelihood[(feature, label)])

    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    return logJoint

  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2)

    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
    for feature in self.features:
        tpl = (self.likelihood[(feature, label1)] / self.likelihood[(feature, label2)], feature)
        featuresOdds.append(tpl)
    featuresOdds.sort()
    return [feature for value, feature in featuresOdds[-100:]]
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()


def _test():
    pass
if __name__ == "__main__":
  _test()


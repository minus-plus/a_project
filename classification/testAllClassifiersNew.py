import dataClassifier
import samples
import sys
from sklearn.externals import joblib

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70
TEST_SET_SIZE = 50

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """
def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
    from optparse import OptionParser
    parser = OptionParser(USAGE_STRING)
    parser.add_option('-d', '--testDigits', help=default("Amount of digits test data to use"), default=TEST_SET_SIZE, type="int")
    parser.add_option('-f', '--testFaces', help=default("Amount of digits test data to use"), default=TEST_SET_SIZE, type="int")
    options, otherjunk = parser.parse_args(argv)

    return options

options = readCommand( sys.argv[1:] )
numTestD = options.testDigits
numTestF = options.testFaces
#==================
#test for faces
#==================
printImage = dataClassifier.ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
rawTestData = samples.loadDataFile("data_CS520/facedatatestn", numTestF,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
featureFunction = dataClassifier.enhancedFeatureExtractorFace
testLabels = samples.loadLabelsFile("data_CS520/facedatatestlabels", numTestF)
testData = map(featureFunction, rawTestData)


print "Testing naiveBayes for faces detection..."
classifier = joblib.load('/tmp/naiveBayes_faces_classifier.joblib.pkl')
guesses = classifier.classify(testData)
correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
dataClassifier.analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

print "\nTesting perceptron for faces detection..."
classifier = joblib.load('/tmp/perceptron_faces_classifier5.joblib.pkl')
guesses = classifier.classify(testData)
correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
dataClassifier.analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

print "\nTesting svm for faces detection..."
classifier = joblib.load('/tmp/svm_faces_classifier.joblib.pkl')
guesses = classifier.classify(testData)
correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
dataClassifier.analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

#==================
#test for digits
#==================
printImage = dataClassifier.ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
rawTestData = samples.loadDataFile("data_CS520/testimagesn", numTestD,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
featureFunction = dataClassifier.enhancedFeatureExtractorDigit
testLabels = samples.loadLabelsFile("data_CS520/testlabels", numTestD)
testData = map(featureFunction, rawTestData)


print "\nTesting naiveBayes for digits recognition..."
classifier = joblib.load('/tmp/naiveBayes_digits_classifier.joblib.pkl')
guesses = classifier.classify(testData)
correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
dataClassifier.analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

print "\nTesting perceptron for digits recognition..."
classifier = joblib.load('/tmp/perceptron_digits_classifier.joblib.pkl')
guesses = classifier.classify(testData)
correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
dataClassifier.analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)

print "\nTesting svm for digits recognition..."
classifier = joblib.load('/tmp/svm_digits_classifier.joblib.pkl')
guesses = classifier.classify(testData)
correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
dataClassifier.analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)


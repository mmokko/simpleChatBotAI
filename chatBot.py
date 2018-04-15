import numpy
import scipy.optimize as optimize
import operator
from nltk import PorterStemmer


def sigmoid(z):
    return 1 / (1 + (numpy.exp(-z)));


class LearningAlgorithm(object):
    def __init__(self):
        self._X = numpy.zeros(0)
        self._y = numpy.zeros(0)
        self._theta = numpy.zeros(0)
        self._common_words = list()

    @staticmethod
    def findMostCommonWords(trainingData):
        words = dict()
        for input in trainingData.TrainingData:
            for word in input.x.split():
                word = PorterStemmer().stem(word.lower())
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
        sorted_words = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
        return [word[0] for word in sorted_words][:trainingData.n]

    def _findFeatureVector(self, trainingData):
        self.findMostCommonWords(trainingData)
        for i in range(0, trainingData.m):
            x = numpy.zeros(trainingData.n)
            for word in trainingData.TrainingData[i].x.split():
                word = PorterStemmer().stem(word.lower())
                if word in self._common_words:
                    x[self._common_words.index(word)] = 1
            self._X[i,:] = x

    def _createTrainingInputMatrix(self, trainingData):
        self._X.resize(trainingData.m, trainingData.n)
        self._common_words = self.findMostCommonWords(trainingData)
        self._findFeatureVector(trainingData)

    def _createTrainingOutputVector(self, trainingData):
        self._y.resize(trainingData.m)
        for i in range(0, trainingData.m):
            y = trainingData.TrainingData[i].y
            self._y[i] = y

    @staticmethod
    def calculateCost(theta, X, y):
        m, n = X.shape
        s = sigmoid(X.dot(theta))
        v1 = numpy.dot(-y, numpy.log(s))
        v2 = (1 - y).dot(numpy.log(1 - s))
        return 1 / m * numpy.sum(v1 - v2)

    @staticmethod
    def calculateGradient(theta, X, y):
        m, n = X.shape
        s = sigmoid(X.dot(theta))
        grad = numpy.zeros(m)
        grad = 1 / m * (X.T.dot(numpy.transpose(s - y)))
        return grad.flatten()

    def optimize(self, trainingData):
        self._theta.resize(trainingData.n)
        self._createTrainingInputMatrix(trainingData)
        self._createTrainingOutputVector(trainingData)
        self._theta = optimize.minimize(
            fun=self.calculateCost,
            x0=self._theta,
            args=(self._X, self._y),
            method='BFGS',
            jac=self.calculateGradient
        ).x

    def _createFeatureVect(self, input):
        x = numpy.zeros(len(self._common_words))
        for i in input:
            for word in input.x.split():
                word = PorterStemmer().stem(word.lower())
                if word in self._common_words:
                    x[self._common_words.index(word)] = 1
        return x

    def blocked(self, input):
        feature_vect = self._createFeatureVect(input)
        prob = sigmoid(feature_vect.dot(self._theta))
        return True if prob >= 0.5 else False

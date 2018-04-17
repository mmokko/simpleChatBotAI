import numpy
import scipy.optimize as optimize
import operator
from nltk import PorterStemmer

# The sigmoid function is used to map the output of our 
# prediction z = x * theta into a probability value (range [0, 1])
def sigmoid(z):
    return 1 / (1 + (numpy.exp(-z)));


class LearningAlgorithm(object):
    def __init__(self):
        self._X = numpy.zeros(0)        # Matrix (m x n) - these will be the m rows of input features for training 
                                        #                  (1 = , 0 = keyword not present)
        self._y = numpy.zeros(0)        # Vector (m) - these will be the m results (1 = blocked, 0 = not blocked)
        self._theta = numpy.zeros(0)    # Vector (n) - these will be the coefficients that we optimise
        self._common_words = list()     # These will be the list on n words that we will use as features

    # This function finds the n most common words on the training data :) 
    # These "common words" are our feature set.
    # What this function does is
    # 1) Gets on sentence in training data
    # 2) Separates each sentence into words
    # 3) Normalises word by lowercasing and stemming (see note below)
    # 4) Counts occurences of each word in a dictionary of words
    # 5) Does this for all words and all sentences in TrainingData
    # 6) Returns the n normalised words with the highest count
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

    # This fills-in the input feature matrix X(m,n) from the training data
    # 1) Expects _common_words has already been filled in with the
    #    n most common normalised words from the training set
    # 2) Then it goes through each of the m inputs of the training set and
    # 2.1) splits each sentence into words
    # 2.2) normalises each word by lowercasing and stemming (see note below)
    # 2.3) if normalised word is found on the common words set, on index j, 
    #      X(i, j) becomes 1, otherwise it is zero
    #
    # So at the end of all this, we have matrix X(m, n) filled with 0 and 1
    # representing the m sentences of the training set, and for each sentence
    # the information whether each of the n normalised common words was present 
    # on the sentence or not
    def _findFeatureVector(self, trainingData):
        for i in range(0, trainingData.m):
            x = numpy.zeros(trainingData.n)
            for word in trainingData.TrainingData[i].x.split():
                word = PorterStemmer().stem(word.lower())
                if word in self._common_words:
                    x[self._common_words.index(word)] = 1
            self._X[i,:] = x

    # This function takes all the steps to fill in the matrix X from the training data
    # 1) dimension to right size (m, n)
    # 2) fill in _common_words to be used in next step
    # 3) call _findFeatureVector to get X filled with 0 and 1's (see explanation above)
    def _createTrainingInputMatrix(self, trainingData):
        self._X.resize(trainingData.m, trainingData.n)
        self._common_words = self.findMostCommonWords(trainingData)
        self._findFeatureVector(trainingData)

    # This function fills in the vector y from the training data, quite simply 
    # copies whether the sentence was to be blocked (1) or not (0)
    def _createTrainingOutputVector(self, trainingData):
        self._y.resize(trainingData.m)
        for i in range(0, trainingData.m):
            y = trainingData.TrainingData[i].y
            self._y[i] = y

    # This is the cost function that we are trying to minimise.
    # Basically it is used to compute "how far" our current model parameters (theta) 
    # are from predicting the values (y) on the training data (X). The 
    # perfect match would be when cost is zero.
    # For more info about this equation, check "cross-entropy" or "log loss" 
    # cost function
    @staticmethod
    def calculateCost(theta, X, y):
        m, n = X.shape
        s = sigmoid(X.dot(theta))
        v1 = numpy.dot(-y, numpy.log(s))
        v2 = (1 - y).dot(numpy.log(1 - s))
        return 1 / m * numpy.sum(v1 - v2)

    # This is the gradient function used to "step" our model parameters (theta) 
    # to get us closer to an optimal point (the function we are trying to mimimise)
    # It uses the derivative of the cost function above
    @staticmethod
    def calculateGradient(theta, X, y):
        m, n = X.shape
        s = sigmoid(X.dot(theta))
        grad = numpy.zeros(m)
        grad = 1 / m * (X.T.dot(numpy.transpose(s - y)))
        return grad.flatten()

    # This is the method that puts everything together to do the learning. 
    # 1) Resize our model parameters vector theta (n)
    # 2) Fill in features matrix X (m, n) from training data
    # 3) Fill in results vector Y (m) from training data
    # 4) Calculate theta such that 
    #        Y ~ X * theta
    #    using the provided cost and gradient functions and the 
    #    BFGS algorithm for minimisation through iteration
    #
    # For more info check SciPy documentation and BGFS algorithm
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

    # This function here is used to map the features (keywords) from a 
    # given (new) sentence into a vector x(n) so that it can be used with 
    # the calculated model parameters to calculate the probability
    # of it been a "blocked" sentence or not. It assumes _common_words has
    # already been filled in by the training part.
    # 1) initialise x to n zeros
    # 2) lower case and stem each word on the input sentence
    # 3) if the normalised word is present on the _common_words, on index j
    #    then x(j) becomes 1
    def _createFeatureVect(self, input):
        x = numpy.zeros(len(self._common_words))
        for i in input:
            for word in input.x.split():
                word = PorterStemmer().stem(word.lower())
                if word in self._common_words:
                    x[self._common_words.index(word)] = 1
        return x

    # Finally, this is the function that decides if a sentence should be 
    # blocked or not. 
    # 1) calculates the feature vector x(n) that represents the sentence 
    # 2) calculates the vector product x*theta
    # 3) uses the sigmoid function to map the calculated value into a probability
    # 4) returns true if the probability >= 0.5 otherwise returns false :)
    def blocked(self, input):
        feature_vect = self._createFeatureVect(input)
        prob = sigmoid(feature_vect.dot(self._theta))
        return True if prob >= 0.5 else False


# Note: normalising the words putting them all lowercase and using a stemmer, 
#       makes the most out of a small data set by clumping together (into the 
#       same feature) similar words like: WORK, working, Worker...
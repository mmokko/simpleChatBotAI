# simpleChatBotAI

## How it all works
Based on email spam classifier algorithm.

1. Find out what your features are.
2. Create training set.
3. Create cross validation set.
4. Create test set.
5. Decide what algorithm should be used.
   * Better to start with quick algorith to test how it all works.
6. Run your algorithm with training date to get as small as possible cost function.
7. Run againt cross validation set.
   * Optimaze algorithm using cross validation resuts.
   * Iterate.
8. Test againt test data.

### Finding features (inputs)

Features x: Choose n amount of words indicative of sentence to block / not block.  
Find most frequently occuring words in the training set. This is your feature vector.
Optionally normalize the data:
* Use stemmer software (Porter Stemmer).  
  For example word "story or stories" would be normalized to "stori"  
```python
from nltk import PorterStemmer
word = PorterStemmer.stem("stories")
```
* Normalize all words to lowercase.
* Fix misspellings.

Create feature vector from received sentence:  
Vector contains boolean value for all blocked words, so we get vector nx1.  
If blocked word appears in the sentence then the boolean value in the feature vector would be set to true (1).  
Example:  
Blocked words: "work", "git", "story", "backlog".  
Sentence: "I added new story to our backlog."  
Feature vector:  

	[0, 0, 1, 1]  

This is the input value to machine learning algorithm.  

### Variables
x = features of email  
y = boolean classifier: block (1) or don't block (0)  
n = number of features  
m = number of training examples  
theta =  weight variable

## Algorithm
### Initializing the algorithm.
1. Create vocabulary list.
   * Find out 10 most used words in training data.
   * Every word has an index in the list.
2. Create matrix X from training data.
   * Feature vector for all training examples (matrix of mxn).
3. Create classifier vector y (vector mx1)
4. Calculate cost function and gradient
   * Initialize Theta to random values.
5. Calculate optimal theta using minimize function.
   * For example in python use scipy and Newton Conjugate Gradient.

### Using the algorithm
1. Create feature vector of the received message.
2. Calculate y using optimal theta.
3. Create predict function to decide if the output is positive or negative.
   * Output value between 0..1 should be changed to binary output true or false.

## Examples
[Spam assassin training data](http://spamassassin.apache.org/old/publiccorpus/)



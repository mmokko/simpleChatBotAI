from collections import namedtuple

Input = namedtuple('Input', ['x', 'y'])

TrainingData = [
    Input("I added a new git repository to github", 1),
    Input("Checkout the git repository first", 1),
    Input("I will create a new story for this sprint", 1),
    Input("The feature is to create new AI algorithm for detecting spam", 1),
    Input("Name of the branch in git is AI chat bot", 1),
    Input("Should we go through all stories in the backlog", 1),
    Input("What time the backlog grooming meeting starts", 1),
    Input("What stories should we estimate today", 1),
    Input("Can you close the story by the end of the sprint", 1),
    Input("Good morning how are you", 0),
    Input("Weather is nice today", 0),
    Input("My name is Mike", 0),
    Input("Liverpool is the best foorball club in the world", 0),
    Input("How are you doing today", 0),
    Input("How was your weekend", 0),
    Input("When is your holiday starting", 0)
]

CrossValidationData = [
    Input("I closed the story i was working on this sprint", 1),
    Input("What was the repository name?", 1),
    Input("Sun is shining today", 0),
    Input("What is your name?", 0)
]

# TODO
# TestData = []

n = 10 # size of the feature vector
m = len(TrainingData)

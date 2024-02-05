# Frank Rosenblatt 1957
# Idea - Perceptron could simulate brain principles with the ability
# to learn and make decisions.  The original Perception was designed to 
# take a number of binary inputs and produce one binary output.  The idea
# was to use weights to represent the importance of each input.  The sum of 
# the values (input*weight) should be greater than a threshold before making a decision
# like yes or no.  The Rosenblatt algorithm was to set a threshold value, 
# multiply all inputs with its weights, sum all the results and activate
# the output.  
#
# More generally, Perceptron is an algorithm for supervised learning of binary classifiers.  
# A binary classifer is a function that can decide whether or not an input, 
# represented by a vector of numbers, belongs to some specific class.
# It is a type of linear classifier that makes its predictions based on a 
# linear predictor function combining a set of weights with the feature vector.
# Perceptorn defines the first step in Nueural Networks
#
# Terminology
# Perceptron Inputs (nodes). 
# Nodes have a value and a weight
# Activate Function (sum > theshold)
#
#imports
import numpy as np
import matplotlib.pyplot as plt

# globals

# data
numPoints = 500

# Perceptron variables
numInputs = 2    # x,y for a point
learnc = 0.00001 
bias = 1
# Line Function y = mx + b for the classifer to learn
def f(x):
    return (x * 1.2) + 50



# Create data - random xy points on a 400x400 graph
xMin = 0
yMin = 0
xMax = 400
yMax = 400
xPoints = np.zeros(numPoints, dtype=int)
yPoints = np.zeros(numPoints, dtype=int)
colors = np.empty(numPoints, dtype=str)
i = 0
while i < numPoints:
    xPoints[i] = np.random.randint(xMin, xMax+1)
    yPoints[i] = np.random.randint(yMin, yMax+1)
    i = i + 1
 
 
# compute desired answer for each point
# 1 if a point is above the line in the graph
# 0 if a point is below or on the line
desired = np.zeros(numPoints, dtype=int)
i = 0
while i < numPoints:
    if yPoints[i] > f(xPoints[i]):
        desired[i] = 1
    i = i + 1

# seed random weights that will be adjusted on each training cycle
weights = np.zeros(numInputs, dtype=float)
i = 0
while i < numInputs:
    randno = np.random.uniform(-1, 1)
    weights[i] = randno
    i = i + 1

# Activate Function (threshold function)
# Binary classifier 
# Maps it input x to and output f(x)
# f(x) = 1 if wx + b > 0; else 0
# where w is a vector of real-valued weights and wx is the dot product
# sum w(i)(x(i) for m inputs and b is the bias
#
def activate(xPoint, yPoint, weightX, weightY):
    sum = 0
    sum = (xPoint * weightX) + (yPoint * weightY)
    sum += bias
    if sum > 0:
        return 1
    else:    
        return 0

# Training Function
def train(xPoint, yPoint, desired):
    guess = activate(xPoint, yPoint, weights[0], weights[1])
    error = desired - guess
    if error != 0:
        weights[0] += learnc * error * xPoint
        weights[1] += learnc * error * yPoint

# Train the Perceptron
i = 0
numTrainingIterations = 10000
while i < numTrainingIterations:
    j = 0
    while j < numPoints:
        train(xPoints[j], yPoints[j], desired[j])
        j = j + 1
    i = i + 1

 # Print the result
i = 0
while i < numPoints:  
    best_guess = activate(xPoints[i], yPoints[i], weights[0], weights[1]) 
    if best_guess == 0:
       colors[i] =  'b'
    else:
        colors[i] = 'r'
    i = i + 1
plt.scatter(xPoints, yPoints, color=colors)
xpoints = np.array([xMin, xMax])
ypoints = np.array([f(xMin), f(xMax)])
plt.plot(xpoints, ypoints)
plt.show



    





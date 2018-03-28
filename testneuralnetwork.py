import numpy as np
import random, math, time, threading
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation as animation

np.random.seed(3)
e=math.e
lossDatas = []

#Matrixes of neurons will be a ROW matrix
#Matrixes of weights will be a matrix with dimensions rows=SizeOfLastLayer X columns=SizeOfNextLayer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def differential_of_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

class Thread:
    def __init__(self, func):
        self.t = threading.Thread(target=func,args=[])
        self.t.daemon = True
        self.t.start()
        
class NeuralNetwork:
    def __init__(self,inputSize,hiddenLayerSizes,outputSize):
        self.learningRate = 1
        self.layers = []
        self.input = 2 * np.random.randn(inputSize,1) + 3
        self.layers.append(self.input)
        for layerSize in hiddenLayerSizes:
            self.layers.append(np.random.randn(layerSize,1)+.5)
        self.output = 2 * np.random.randn(outputSize,1) + .5
        self.layers.append(self.output)
        self.biases=[]
        for layer in self.layers[:len(self.layers)-1]:#For every layer that isn't the output layer
            rows,columns = layer.shape
            self.biases.append(np.zeros((1,rows)))#Make a bias that exists parallel to that layer
        self.makeWeights()

    def makeWeights(self):
        self.weights = []
        for index, layer in enumerate(self.layers[:len(self.layers)-1]):#For every layer that isn't the output layer
            rows,columns = layer.shape#Size of that layer
            rows2,columns2 = self.layers[index+1].shape#Size of next layer
            self.weights.append(np.random.randn(rows,rows2)+2)

    def setInputs(self, matrix):
        self.input = matrix
        self.layers[0]=self.input

    def FeedForward(self):
        for index in range(len(self.layers)-1):
            self.layers[index+1] = sigmoid(np.dot(self.layers[index], self.weights[index]))# + self.biases[index])

    def correctWeights(self, layerIndex, lastLayerDelta):
        newDelta = lastLayerDelta.dot(self.weights[-(layerIndex-1)].T)
        #print("newDelta: " + str(newDelta))
        
        newDelta = newDelta * differential_of_sigmoid(self.layers[-layerIndex]) * self.learningRate
        if layerIndex < len(self.layers)-1:
            self.correctWeights(layerIndex+1, newDelta)
        self.weights[-layerIndex] -= newDelta

    def train(self, input, output):
        self.setInputs(input)
        self.FeedForward()

        error = output - self.layers[-1]
        #print("original error: " + str(error))
        outputDelta = self.layers[-1].T.dot(error) * differential_of_sigmoid(output)
        print(outputDelta)
        lossDatas.append(np.abs(np.sum(outputDelta**2)))
        #print("differential of sigmoid: " +str(differential_of_sigmoid(output)))
        self.weights[-1] -= outputDelta

        self.correctWeights(2, outputDelta)
        return outputDelta
     
test = NeuralNetwork(1,[3],1)
thenum = random.randint(1,10)
error = test.train(np.array([[thenum]]),np.array([[int(thenum<5)]]))
time.sleep(.5)
#plot = plt.plot(lossDatas)
#plt.pause(.25)

def plotLoop():
    while True:
        plt.plot(lossDatas)
        plt.pause(.1)
        plt.clf()
        plt.cla()

plotThread = Thread(plotLoop)

while True:
    thenum = random.randint(1,10)
    error = test.train(np.array([[thenum]]),np.array([[int(thenum<5)]]))
    if error**2 < .001:
        #break
        pass



test.setInputs(np.array([[3]]))
test.FeedForward()
print(test.weights[-1])

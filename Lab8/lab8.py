from random import random, shuffle
from copy import deepcopy


def identical(x):
    return x


def dIdentical(x):
    return 1


def linear(x):
    return 0.4*x


def dLinear(x):
    return 0.4


class Neuron:
    def __init__(self, noOfInputs, activationFunction):
        self.noOfInputs = noOfInputs
        self.activationFunction = activationFunction
        self.weights = [random() for i in range(self.noOfInputs)]
        self.output = 0

    def fireNeuron(self, inputs):
        u = sum([x*y for x, y in zip(inputs, self.weights)])
        self.output = self.activationFunction(u)
        return self.output


class Layer:
    def __init__(self, noOfInputs, activationFunction, noOfNeurons):
        # we create the neurons
        # @noOfInputs are the given data (nr of attributes)
        # @activationFunction is in our case Linear Function
        # @noOfNeurons - the nr of neurons from this layer
        self.noOfNeurons = noOfNeurons
        self.neurons = [Neuron(noOfInputs, activationFunction) for i in range(self.noOfNeurons)]

    def forward(self, inputs):
        # here we have a line of data
        # and we want to get the output of each neuron w.r.t. this data
        for x in self.neurons:
            x.fireNeuron(inputs)
        return([x.output for x in self.neurons])


class FirstLayer(Layer):
    # this is the input layer, the weights don't matter
    # the activation function is just an identical function
    def __init__(self, noOfNeurons):
        Layer.__init__(self, 1, identical, noOfNeurons)
        for x in self.neurons:
            x.weights = [1]

    def forward(self, inputs):
        return inputs


class Network:
    def __init__(self, structure, activationFunction, derivate):
        self.activationFunction = activationFunction
        self.derivate = derivate
        # structure is a list with elements. Those represent the
        # number of neurons in each layer
        self.structure = structure[:]
        self.noLayers = len(self.structure)
        self.layers = [FirstLayer(self.structure[0])]
        for i in range(1, len(self.structure)):
            self.layers = self.layers + [Layer(self.structure[i-1],activationFunction,self.structure[i])]

    def feedForward(self, inputs):
        self.signal = inputs[:]
        for l in self.layers:
            self.signal = l.forward(self.signal)
        return self.signal

    def backPropag(self, loss, learnRate):
        err = loss[:]
        delta = []
        currentLayer = self.noLayers-1
        newConfig = Network(self.structure, self.activationFunction, self.derivate)
        # last layer
        for i in range(self.structure[-1]):
            delta.append(err[i])
            for r in range(self.structure[currentLayer-1]):
                newConfig.layers[-1].neurons[i].weights[r] = self.layers[-1].neurons[i].weights[r] + learnRate*delta[i]*self.layers[currentLayer-1].neurons[r].output
        # propagate the errors layer by layer
        for currentLayer in range(self.noLayers-2, 0, -1):
            currentDelta = []
            for i in range(self.structure[currentLayer]):
                currentDelta.append(sum([self.layers[currentLayer+1].neurons[j].weights[i]*delta[j] for j in range(self.structure[currentLayer+1])]))
            delta = currentDelta[:]
            for i in range(self.structure[currentLayer]):
                for r in range(self.structure[currentLayer-1]):
                    newConfig.layers[currentLayer].neurons[i].weights[r] = self.layers[currentLayer].neurons[i].weights[r] + learnRate*delta[i]*self.layers[currentLayer-1].neurons[r].output
        self.layers = deepcopy(newConfig.layers)

    def computeLoss(self, u, t):
        loss = []
        out = self.feedForward(u)
        for i in range(len(t)):
            loss.append(t[i]-out[i])
        return loss


def main():
    nn = Network([5, 5, 1], linear, dLinear)
    repo = Repository()
    repo.getDataFromFile("leastsquare.txt")
    u = repo.dataTrainX
    t = repo.dataTrainY

    for i in range(400):
        for j in range(len(u)):
            nn.backPropag(nn.computeLoss(u[j],t[j]), 0.25)
    k = 0
    trials = 0
    for j in range(len(u)):
        trials += 1
        error = nn.computeLoss(u[j],t[j])
        if (error[0]**2) < 0.1 : k += 1
    print(k/trials)


class Repository:
    def __init__(self, dTrainX=[], dTrainY=[], dTestX=[], dTestY=[]):
        self.dataTrainX = dTrainX
        self.dataTestX = dTestX
        self.dataTrainY = dTrainY
        self.dataTestY = dTestY

    def getDataFromFile(self, filepath):
        fileDescriptor = open(filepath, "r")
        self.d = []
        for line in fileDescriptor:
            attributes = line.split(" ")
            listOfAttributes = []
            for attribute in attributes:
                listOfAttributes.append(float(attribute.strip()))
            self.d.append(listOfAttributes)
        shuffle(self.d)
        self.normalizeData()
        for i in range(int(len(self.d))):
            self.dataTrainX.append(self.d[i][:len(self.d[0])-1])
            self.dataTrainY.append([self.d[i][-1]])

    def normalizeData(self):
        vmax = 0
        vmin = 2000000000
        for i in range(len(self.d)):
            for j in range(len(self.d[i])):
                if vmax < self.d[i][j] : vmax = self.d[i][j]
                if vmin > self.d[i][j] : vmin = self.d[i][j]
        for i in range(len(self.d)):
            for j in range(len(self.d[i])):
                self.d[i][j] = (self.d[i][j]-vmin)/vmax


main()

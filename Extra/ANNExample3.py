from random import random


class Neuron:
    def __init__(self, noOfInputs):
        self.noOfInputs = noOfInputs
        self.weights = [random() for i in range(self.noOfInputs)]
        self.output = 0

    def activateNeuron(self, inputs):
        s = 0
        for i in range(len(inputs)):
            s += inputs[i]*self.weights[i]
        self.output = self.activationFunction(s)
        return self.output

    def activationFunction(self, x):
        return 0.4*x


class Network:
    def __init__(self, nrNeurons, noOfInputs):
        # neurons for hidden layer
        self.nrNeurons = nrNeurons
        self.noOfInputs = noOfInputs
        
        self.neurons = [Neuron(noOfInputs) for i in range(nrNeurons)]
        self.outputneuron = Neuron(nrNeurons)
    
    def forward(self, inputs):
        output = []
        for x in self.neurons:
            output.append(x.activateNeuron(inputs))
        return self.outputneuron.activateNeuron(output)

    def backPropag(self, err, learnRate, inputs):
        currentLayer = 2
        # last layer
        delta = []
        for i in range(self.nrNeurons):
            delta.append(self.outputneuron.weights[i]*err)

        for i in range(1):
            for r in range(self.nrNeurons):
                k = self.outputneuron.weights[r]
                self.outputneuron.weights[r] += learnRate*err*self.neurons[r].output
                p = self.outputneuron.weights[r]

        currentLayer = 1

        for i in range(self.nrNeurons):
            for r in range(len(inputs)):
                k = self.neurons[i].weights[r]
                self.neurons[i].weights[r] += learnRate*delta[i]*inputs[r]
                p = self.neurons[i].weights[r]

    def computeLoss(self, u, t):
        out = self.forward(u)
        return t[0]-out


def test2():
    nn = Network(1,5)
    repo = Repository()
    repo.getDataFromFile("leastsquare.txt")
    u = repo.dataTrainX[:30]
    t = repo.dataTrainY[:30]

    for i in range(100):
        for j in range(len(u)):
            l = nn.computeLoss(u[j],t[j])
            print("ese:", u[j], t[j], l)
            nn.backPropag(nn.computeLoss(u[j], t[j]), 0.1,u[j])

    for j in range(len(u)):
        print(u[j], t[j], nn.feedForward(u[j]))


class Repository:
    def __init__(self, dTrainX=[], dTrainY=[], dTestX=[], dTestY=[]):
        self.dataTrainX = dTrainX
        self.dataTestX = dTestX
        self.dataTrainY = dTrainY
        self.dataTestY = dTestY

    def getDataFromFile(self, filepath):
        fileDescriptor = open(filepath, "r")
        d = []
        for line in fileDescriptor:
            attributes = line.split(" ")
            listOfAttributes = []
            for attribute in attributes:
                listOfAttributes.append(float(attribute.strip()))
            d.append(listOfAttributes)
        shuffle(d)
        for i in range(int(len(d)*0.8)):
            self.dataTrainX.append(d[i][:len(d[0])-1])
            self.dataTrainY.append([d[i][-1]])
        for i in range(int(len(d)*0.8), len(d)):
            self.dataTestX.append(d[i][:len(d[0])-1])
            self.dataTestY.append([d[i][-1]])


test2()
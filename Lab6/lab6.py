'''
Petcu Dragos
925/2
'''
import math
from copy import deepcopy
from random import shuffle


class Node:

    def __init__(self, column, value, leaf):
        self.leaf = leaf  # true is the node is a leaf
        self.column = column  # attribute selected
        self.value = value
        self.children = []

    def addNode(self, node):
        self.children.append(node)


class Repository:
    def __init__(self, data=[], attributes=["C", "LW", "LD", "RW", "RD"]):
        self.data = data
        self.attributes = attributes

    def getDataFromFile(self, filepath, training=500, test=125):
        fileDescriptor = open(filepath, "r")
        d = []
        self.dataTest = []
        for line in fileDescriptor:
            attributes = line.split(",")
            listOfAttributes = []
            for attribute in attributes:
                listOfAttributes.append(attribute.strip())
            d.append(listOfAttributes)
        shuffle(d)
        self.dataTest = d[training:]
        self.data = d[:training]


class Controller:
    def __init__(self, repo):
        self.repo = repo

    def getClassValuesWithFrequencies(self, data):
        values = {}
        for line in data:
            if line[0] not in values:
                values[line[0]] = 1
            else:
                values[line[0]] += 1
        return values

    def getAttributeFrequencyOnIndex(self, data, index, attribute):
        # given an index and an attribute
        # returns the number of appeareances of that attribute in that index
        s = 0
        for line in data:
            if line[index] == attribute:
                s += 1
        return s

    def getAtributeClassFrequency(self, data, index, attribute, clas):
        # given an atribute from an index and a class
        # returns the number of appeareances of that attribute-class tuple
        s = 0
        for line in data:
            if line[index] == attribute and line[0] == clas:
                s += 1
        return s

    def getAttributesFromIndex(self, data, index):
        # given an index and the table
        # returns the possible attributes of that index
        attributes = []
        for line in data:
            if line[index] not in attributes:
                attributes.append(line[index])
        return attributes

    def getDataForAttributeAndIndex(self, data, attribute, index):
        newData = []
        for line in data:
            if line[index] == attribute:
                newLine = deepcopy(line)
                newLine.pop(index)
                newData.append(newLine)
        return newData

    def getBestIndex(self, data):
        classValues = self.getClassValuesWithFrequencies(data)
        if len(classValues.keys()) == 1:
            return [True, list(classValues.keys())[0]]
        S = sum(classValues.values())
        Es = sum(classValues[i]/S*math.log(S/classValues[i], 2) for i in classValues)
        gains = []
        for index in range(1, len(data[0])):
            # we have a list of attributes(values) for this index
            gain = Es
            attributes = self.getAttributesFromIndex(data, index)
            for j in range(len(attributes)):
                Ep = 0
                r = self.getAttributeFrequencyOnIndex(data, index, attributes[j])
                for i in classValues.keys():
                    s = self.getAtributeClassFrequency(data,index,attributes[j],i)
                    if r != 0 and s != 0:
                        Ep += s/r*math.log(r/s, 2)
                ts = Ep*r/len(data)
                gain -= ts
            gains.append([index, gain])
        bestGain = min(gains, key=lambda x: x[1])
        return [False, bestGain[0]]

    def generate(self, repo, attribute):
        data = repo.data
        if len(data[0]) == 1:  # no more attributes
            classes = self.getClassValuesWithFrequencies(data)
            dominantClass = max(classes, key=lambda x: classes[x])
            return Node(dominantClass, attribute, True)
        classMajority, index = self.getBestIndex(data)
        if classMajority:
            return Node(index, attribute, True)
        attributes = ['1', '2', '3', '4', '5']
        field = repo.attributes[index]
        n = Node(field, attribute, False)
        for i in range(len(attributes)):
            newData = self.getDataForAttributeAndIndex(data, attributes[i], index)
            if len(newData) == 0:
                classes = self.getClassValuesWithFrequencies(data)
                dominantClass = max(classes, key=lambda x: classes[x])
                n.addNode(Node(dominantClass, attributes[i], True))
            else:
                newIndexes = deepcopy(repo.attributes)
                newIndexes.pop(index)
                newRepo = Repository(newData, newIndexes)
                n.addNode(self.generate(newRepo, attributes[i]))
        return n

    def test(self, node, line):
        realValue = line[0]
        attributes = ["C", "LW", "LD", "RW", "RD"]
        leaf = False
        while(not leaf):
            for nod in node.children:
                if line[attributes.index(node.column)] == nod.value:
                    node = deepcopy(nod)
                    leaf = nod.leaf
                    break
        if realValue == node.column:
            return 1
        return 0


def main():
    r = 0
    for i in range(75):
        repo = Repository()
        repo.getDataFromFile("balance-scale.data")
        controller = Controller(repo)
        root = controller.generate(repo, 0)
        dataTest = repo.dataTest
        s = 0
        k = 0
        for line in dataTest:
            s += controller.test(root, line)
            k += 1
        r = max(r, s/k)
    print(r)


main()

'''
Petcu Dragos
925/2
'''
from random import shuffle
import numpy as np


class Repository:
    def __init__(self, dataTrainX=[], dataTrainY=[], dataTestX=[], dataTestY=[]):
        #  [@dataTrainX] - a list of values consisting of xi values
        #  [@dataTrainY] - a list of values consisting of yi values
        #  [@dataTestX] - a list of values consisting of xi values
        #  [@dataTestY] - a list of values consisting of yi values
        self.dataTrainX = dataTrainX
        self.dataTestX = dataTestX
        self.dataTrainY = dataTrainY
        self.dataTestY = dataTestY

    def getDataFromFile(self, filepath):
        fileDescriptor = open(filepath, "r")
        d = []
        for line in fileDescriptor:
            attributes = line.split(" ")
            listOfAttributes = [1]
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


class Controller:
    def __init__(self, repository):
        #  repository - Repository type
        self.repository = repository

    def getTransposeOfMatrix(self, matrix):
        #  matrix - npArray
        #  returns the transpose matrix X^T
        return matrix.transpose()

    def multiplyMatrices(self, firstMatrix, secondMatrix):
        #  matrix - npArray
        #  matrix - npArray
        return np.matmul(firstMatrix, secondMatrix)

    def getInverseOfMatrix(self, matrix):
        #  matrix - npArray
        #  returns the inverse matrix X^-1
        return np.linalg.inv(matrix)

    def fx(self, X):
        return self.multiplyMatrices(self.getInverseOfMatrix(self.multiplyMatrices(self.getTransposeOfMatrix(X), X)), self.getTransposeOfMatrix(X))

    def lossFunction(self, fx, y):
        return self.multiplyMatrices(fx, y)

    def leastSquareMethod(self):
        X = np.array(self.repository.dataTrainX)
        Y = np.array(self.repository.dataTrainY)
        Betas = self.lossFunction(self.fx(X), Y)
        return Betas

    def testMethod(self, Betas):
        errors = []
        X = self.repository.dataTestX
        Y = self.repository.dataTestY
        for i in range(len(X)):
            S = 0
            for j in range(len(X[i])):
                S += X[i][j] * Betas[j][0]
            errors.append(abs(Y[i][0] - S))
        return errors


class UserInterface:
    def __init__(self, controller):
        self.controller = controller

    def runLeastSquareMethod(self):
        Betas = self.controller.leastSquareMethod()
        errors = self.controller.testMethod(Betas)
        print("Betas: \n", Betas.tolist())
        print("Error: ", round(np.mean(errors),10))


if __name__ == "__main__":
    repository = Repository()
    repository.getDataFromFile("leastsquare.txt")
    controller = Controller(repository)
    userInterface = UserInterface(controller)
    userInterface.runLeastSquareMethod()

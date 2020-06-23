from random import randint, random
from copy import deepcopy
import itertools
import matplotlib.pyplot as plt
import numpy as np


class Ant:
    def __init__(self, n):
        # @n the size of the matrix
        self.n = n
        # a list from 1 to @n
        symbolList = list(range(1, n+1))
        # a list of valid permutations with list @symbolList having @n elements
        permutation = list(itertools.permutations(symbolList, n))
        # choosing 2 valid permutations to start the matrix with.
        firstPermutation = permutation[randint(0, len(permutation)-1)]
        secondPermutation = permutation[randint(0, len(permutation)-1)]
        # list of tuples
        self.tuples = []
        # constructing the matrix
        self.path = []
        for i in range(n):
            self.path.append([firstPermutation[i]])
            self.path.append([secondPermutation[i]])
            self.tuples.append([firstPermutation[i],secondPermutation[i]])


class Problem:
    def nextMoves(self, ant, index):
        # given an @index of a permutation, returns a list of next moves
        # i.e. the possible next elements in that permutation
        movesList = []
        for i in range(1, ant.n+1):
            if i not in ant.path[index]:
                movesList.append(i)
        return movesList

    def distMove(self, ant, index, element):
        # returns the distance given by adding an element
        dummy = Ant(ant.n)
        dummy.path = deepcopy(ant.path)
        dummy.path[index].append(element)
        if index+2 <= ant.n:
            return ant.n-len(self.nextMoves(dummy, index+2))
        else:
            return ant.n

    def addMove(self, ant, q0, trace, alpha, beta):
        # adds a new move to the ant, if possible
        for index in range(len(ant.path)):
            # 0 means invalid move
            p = [0 for i in range(ant.n)]
            nextSteps = deepcopy(self.nextMoves(ant, index))
            if (len(nextSteps) == 0):
                continue
            # for each move we keep the distance
            for i in nextSteps:
                p[i-1] = self.distMove(ant, index, i)
            # calculate trace^alpha and visibility^beta
            p = [(p[i]**beta)*(trace[index][ant.path[index][-1]-1][i]**alpha) for i in range(len(p))]
            if (random() < q0):
                # get best move
                p = [[i, p[i]] for i in range(len(p))]
                p = max(p, key=lambda a: a[1])
                ant.path[index].append(p[0]+1)
                if index % 2 == 1:
                    ant.tuples.append([ant.path[index-1][-1], ant.path[index][-1]])
            else:
                # adaugam cu o probabilitate un drum posibil (ruleta)
                s = sum(p)
                if (s == 0):
                    ant.path[index].append(nextSteps[randint(0, len(nextSteps)-1)])
                    if index % 2 == 1:
                        ant.tuples.append([ant.path[index-1][-1], ant.path[index][-1]])
                else:
                    p = [p[i]/s for i in range(len(p))]
                    p = [sum(p[0:i+1]) for i in range(len(p))]
                    r = random()
                    i = 0
                    for i in range(len(p)):
                        if(r < p[i]):
                            ant.path[index].append(i+1)
                            if index % 2 == 0:
                                ant.tuples.append([ant.path[index-1][-1], ant.path[index][-1]])
                            break

    def fitness(self, ant):
        f = 1
        for i in range(0, ant.n):
            visited = []
            for j in range(0, ant.n*2, 2):
                if ant.path[j][i] in visited:
                    f += 1
                else:
                    visited.append(ant.path[j][i])
            visited = []
            for j in range(1, ant.n*2, 2):
                if ant.path[j][i] in visited:
                    f += 1
                else:
                    visited.append(ant.path[j][i])
        for i in range(len(ant.tuples)):
            if ant.tuples[i] in ant.tuples[i+1:]:
                f += 1
        return f


class Controller:
    def __init__(self):
        self.problem = Problem()

    def epoca(self, noAnts, n, trace, alpha, beta, q0, rho):
        antSet = [Ant(n) for i in range(noAnts)]
        for i in range(n-1):
            for x in antSet:
                self.problem.addMove(x, q0, trace, alpha, beta)
        dTrace = [1.0 / self.problem.fitness(antSet[i]) for i in range(len(antSet))]
        for i in range(2*n):
            for j in range(n):
                for k in range(n):
                    trace[i][j][k] = (1 - rho) * trace[i][j][k]
        for i in range(len(antSet)):
            for j in range(len(antSet[i].path)):
                for k in range(len(antSet[i].path[j])-1):
                    x = antSet[i].path[j][k]-1
                    y = antSet[i].path[j][k+1]-1
                    trace[j][x][y] = trace[j][x][y] + dTrace[i]
        # return best ant path
        f = [[self.problem.fitness(antSet[i]), i] for i in range(len(antSet))]
        f = max(f)
        return antSet[f[1]]


class Console:
    def __init__(self):
        self.controller = Controller()

    def runProgram(self, n=4, it=300, noAnts=10, alpha=1.9, beta=0.9, rho=0.05, q0=0.5):
        sol = []
        bestSol = []
        bestFitness = 1000000
        trace = [[[1 for i in range(n)] for j in range(n)] for k in range(2*n)]
        for i in range(it):
            sol = self.controller.epoca(noAnts, n, trace, alpha, beta, q0, rho)
            if self.controller.problem.fitness(sol) < bestFitness:
                bestSol = deepcopy(sol.path)
                bestFitness = self.controller.problem.fitness(sol)
        return [bestFitness-1, bestSol]

    def runStatistics(self):
        n = 3
        noAnts = 15
        it = 150
        k = 0
        listFitness = []
        while k <= 30:
            k += 1
            fitness, bestSol = self.runProgram(n, it, noAnts)
            listFitness.append(fitness)
            print("Iteration", k, "finished")
        print("\nACO Statistics:")
        meanEA = np.mean(listFitness)
        stdEA = np.std(listFitness)
        print("Mean: ")
        print(meanEA)
        print("Std Deviation: ")
        print(stdEA)
        plt.plot(listFitness, 'ro')

    def run(self):
        option = int(input("1-Run program\n2-Run Statistics\n"))
        if option == 1:
            n = 4
            noAnts = 10
            it = 200
            try:
                n = int(input("n="))
            except Exception:
                print("n = 4")
            try:
                noAnts = int(input("number of ants="))
            except Exception:
                print("noAnts = 10")
            try:
                it = int(input("iterations="))
            except Exception:
                print("iterations = 200")
            bestFitness, bestSol = self.runProgram(n, it, noAnts)
            print(bestFitness, "\n")
            for i in range(0, 2*n-1, 2):
                line = []
                for j in range(n):
                    line.append([bestSol[i][j], bestSol[i+1][j]])
                print(line)
        else:
            self.runStatistics()


def main():
    c = Console()
    c.run()


main()

import sys
from random import random
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QVBoxLayout,QHBoxLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import QThread,pyqtSignal
from random import randint
import threading
from copy import deepcopy
import itertools

class QThread1(QThread):

    sig1 = pyqtSignal(str)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.controller = Controller()

    def setEntities(self,option,trials,n,dimPopulation=0,probMutate=0.2,w=1,c1=1.8,c2=1.2):
        self.trials = trials
        self.n = n
        self.dimPopulation = dimPopulation
        self.probMutate = probMutate
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.option = option
    def run(self):
        if self.option == 1: 
            self.controller.HillClimbing(self.sig1,self.trials,self.n)
        if self.option == 2: 
            self.controller.EvolutionaryAlgorithm(self.sig1,self.trials,self.n,self.dimPopulation,self.probMutate)
        if self.option == 3:
            self.controller.ParticleSwampOptimization(self.sig1,self.trials,self.n,self.dimPopulation,self.w,self.c1,self.c2)
    
    def stop(self):
        self.terminate()
        self.wait()


class UserInterface(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.controller = Controller();
        self.buttonHC = QPushButton("Run Hill Climbing Algorithm!")
        self.buttonEA = QPushButton("Run Evolutionary Algorithm!")
        self.buttonPSO = QPushButton("Run Particle Swarm Optimisation Algorithm!")
        self.buttonStop = QPushButton("Stop!")
        self.labelTrials = QLabel("Number of trials")
        self.textTrials = QLineEdit()
        self.labelDIndividual = QLabel("Dimension of individual")
        self.textDIndividual = QLineEdit()
        self.labelDPopulation = QLabel("Dimension of population")
        self.textDPopulation = QLineEdit()
        self.labelMutation = QLabel("Mutation Probability")
        self.textMutation = QLineEdit()
        self.labelMatrix = QLabel("Initial matrix")        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.labelTrials)
        self.layout.addWidget(self.textTrials)
        self.layout.addWidget(self.labelDIndividual)
        self.layout.addWidget(self.textDIndividual)
        self.layout.addWidget(self.labelDPopulation)
        self.layout.addWidget(self.textDPopulation)
        self.layout.addWidget(self.labelMutation)
        self.layout.addWidget(self.textMutation)
        self.layout.addWidget(self.buttonHC)
        self.layout.addWidget(self.buttonEA)
        self.layout.addWidget(self.buttonPSO)
        self.hlayout = QHBoxLayout()
        self.hlayout.addLayout(self.layout)
        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.labelMatrix)
        self.vlayout.addWidget(self.buttonStop)
        self.vlayout.setContentsMargins(0,0,500,500)
        self.hlayout.addLayout(self.vlayout)
        self.setLayout(self.hlayout)
        self.buttonHC.clicked.connect(self.executeHCAlgorithm)
        self.buttonEA.clicked.connect(self.executeEAAlgorithm)
        self.buttonPSO.clicked.connect(self.executePSOAlgorithm)
        self.buttonStop.clicked.connect(self.stop)
        
        
        self.t = threading.Thread()
        self.thread1 = QThread1()
        self.thread1.sig1.connect(self.printin)
        
    def executeHCAlgorithm(self):
        self.thread1.stop()
        try:
            trials = int(str(self.textTrials.text()))
        except:
            trials = 1000
            
        try:
            n = int(str(self.textDIndividual.text()))
        except:
            n = 4
        self.thread1.setEntities(1,trials,n)
        self.thread1.start()
        
        
    def executeEAAlgorithm(self):
        self.thread1.stop()
        try:
            trials = int(str(self.textTrials.text()))
        except:
            trials = 1000
        try:
            n = int(str(self.textDIndividual.text()))
        except: 
            n = 4
        try:
            dimpop = int(str(self.textDPopulation.text()))
        except: 
            dimpop = 100
        try:   
            probMutation = int(str(self.textMutation.text())) 
        except: 
            probMutation = 0.2
        self.thread1.setEntities(2,trials,n,dimpop,probMutation)
        self.thread1.start()
        
    def executePSOAlgorithm(self):
        self.thread1.stop()
        try:
            trials = int(str(self.textTrials.text()))
        except:
            trials = 1000
        try:
            n = int(str(self.textDIndividual.text()))
        except: 
            n = 4
        try:
            dimpop = int(str(self.textDPopulation.text()))
        except: 
            dimpop = 100
        w = 1
        c1 = 1.5
        c2 = 0.8    
        self.thread1.setEntities(3,trials,n,dimpop,w,c1,c2)
        self.thread1.start()
        
    def stop(self):
        self.thread1.stop()
    def printin(self,s):
        self.labelMatrix.setText(s)

def printOut(s):
    print(s)
def printMatrix(m,trial,trials):
    s = "Trial "+ str(trial) + " from " + str(trials) + " trials \n Fitness level : " + str(m.fitness) + "\n"
    for row in range(len(m.state)):
        s = s + str(m.state[row]) + "\n"
    return s 

class Individual :
    
    def __init__(self, dimIndividual,state=0):
        self.dimIndividual = dimIndividual
        if state == 0: 
            self.state = self.createIndividual()
        else :
            self.state = state
        self.fitness = self.fit()
        self.velocity = self.createVelocity()
        
    
    def createIndividual(self) :
        length = self.dimIndividual
        symbolList = list(range(1,length+1))
        permutations = list(itertools.permutations(symbolList,length))
        k = 0
        m = []
        while len(permutations) > 1 and k < length:
            ind1 = randint(0,len(permutations)-1)
            p1 = permutations.pop(ind1)
            ind2 = randint(0,len(permutations)-1)
            p2 = permutations.pop(ind2)
            mi = []
            for j in range(length) :
                mi.append([p1[j],p2[j]])
            m.append(mi)
            k += 1
        return m
    
    def fit(self):
        f=0;
        individual = self.state
        for i in range(len(individual)):
            for j in range(len(individual[i])):
                for ii in range(i+1,len(individual)):
                    if individual[ii][j][0] == individual[i][j][0] : f+=1
                    if individual[ii][j][1] == individual[i][j][1] : f+=1
        l = []
        for i in range(len(individual)) :
            l.extend(individual[i])
        for i in range(len(l)) :
            if l[i] in l[i+1:] : f += 1
        return f

    def getNeighbourhood(self) :
        stateList = []
        n = self.dimIndividual
        for row in range(len(self.state)) :
            position1 = randint(0,n-1) 
            position2 = randint(0,n-1) 
            position3 = randint(0,n-1) 
            position4 = randint(0,n-1) 
            newState1 = deepcopy(self.state)
            newState2 = deepcopy(self.state)
            q1 = newState1[row][position1][0]
            q2 = newState1[row][position2][0] 
            newState1[row][position2][0] = q1 
            newState1[row][position1][0] = q2
            q3 = newState2[row][position3][1]
            q4 = newState2[row][position4][1]
            newState2[row][position4][1] = q3 
            newState2[row][position3][1] = q4
            stateList.append(Individual(n,newState1))
            stateList.append(Individual(n,newState2))
        return stateList
    
    def createVelocity(self):
        velocity = []
        for i in range(self.dimIndividual):
            velocity.append([0,0])
        return velocity




class Controller :
    
    def HillClimbing(self,signal,trials,n):
        currentState = Individual(n)
        trial = 0
        while currentState.fitness != 0 and trial <= trials:
            newStateList = []
            neighbourhood = currentState.getNeighbourhood()
            for state in neighbourhood:
                newStateList.append(state)
            newStateList.sort(key=lambda x : x.fitness)
            newState = newStateList[0]
            if newState.fitness < currentState.fitness :
                currentState = deepcopy(newState)
            if signal != 0 : signal.emit(printMatrix(currentState,trial,trials))
            else: printOut(printMatrix(currentState,trial,trials))
            trial += 1
        return currentState

    def EvolutionaryAlgorithm(self, signal,iterations,dimIndividual,dimPopulation,probMutation=0.2):
        P = self.population(dimPopulation, dimIndividual)
        result = []
        for i in range(iterations):
            P = self.iteration(P, probMutation)
            P.sort(key=lambda x : x.fitness)
            result=P[0]
            if (P[0].fitness == 0) :
                break
            if signal != 0 : signal.emit(printMatrix(result,i,iterations))
            else: printOut(printMatrix(result,i,iterations))
        return result
      
    
    def ParticleSwampOptimization(self,signal,iterations,dimIndividual,dimPopulation,w,c1,c2):
        P = self.population(dimPopulation, dimIndividual)

        neighbours = []
        for neighbor in P:
            neighbours.append(neighbor.getNeighbourhood())
        result = []
        for i in range(iterations):
            P = self.iterationPSO(P, neighbours, c1, c2, w/(i+1))
            pop = [ x for x in P]
            pop.sort(key=lambda x : x.fitness)
            result=pop[0]
            if signal != 0 : signal.emit(printMatrix(result,i,iterations))
            else: printOut(printMatrix(result,i,iterations))
            if(result.fitness == 0) : break
        return result
    def dManhattan(self,m1,m2,k):
        s = 0
        for i in range(len(m1)):
            s+= abs(m1[i][k]-m2[i][k])
        return s
    
    def iterationPSO(self,population,neighbours,c1,c2,w):
        bestNeighbours = []
        bestOfAll = deepcopy(population[0].state)
        bestOfAllFitness = population[0].fitness
        for i in range(len(population)):
            bestNeighbour = deepcopy(neighbours[i][0])
            bestFitness = bestNeighbour.fitness
            for j in range(1,len(neighbours[i])):
                if neighbours[i][j].fitness < bestFitness :
                    bestNeighbour = deepcopy(neighbours[i][j])
                    bestFitness = bestNeighbour.fitness
                if(neighbours[i][j].fitness < bestOfAllFitness):
                    bestOfAll = deepcopy(neighbours[i][j])
                    bestOfAllFitness = bestOfAll.fitness
            bestNeighbours.append(bestNeighbour)
        
        newpop = []
        m = len(population[0].state)
        for n in range(len(population)) :
            newind = []
            for i in range(m):
                newVelocityX = population[n].velocity[i][0]*w
                newVelocityY = population[n].velocity[i][1]*w
                newVelocityX = newVelocityX + c1*random()*self.dManhattan(bestNeighbours[n].state[i],population[n].state[i],0)    
                newVelocityX = newVelocityX + c2*random()*self.dManhattan(bestOfAll.state[i],population[n].state[i],0)
                newVelocityY = newVelocityY + c1*random()*self.dManhattan(bestNeighbours[n].state[i],population[n].state[i],1)    
                newVelocityY = newVelocityY + c2*random()*self.dManhattan(bestOfAll.state[i],population[n].state[i],1)
                population[n].velocity[i][0] = newVelocityX
                population[n].velocity[i][1] = newVelocityY
                l = []
                if newVelocityX > random() :
                    if(newVelocityY > random()) :
                        l = deepcopy(bestNeighbours[n].state[i])
                    else :
                        for j in range(m):
                            l.append([bestNeighbours[n].state[i][j][0],population[n].state[i][j][1]])
                else:
                    if(newVelocityY > random()) :
                        for j in range(m):
                            l.append([population[n].state[i][j][0],bestNeighbours[n].state[i][j][1]])
                    else :
                        l = population[n].state[i]
                newind.append(l)
            newpop.append(newind)
                       
        for n in range(len(population)):
            population[n] = Individual(m,newpop[n])
                
        return population
    
    def iteration(self,pop, pM):
        '''
            so in this iteration we take 2 random matrices and them we crossover
            after that we try to mutate
            we have a result and we compare the fitness to the parents and replace it if better
        '''
        i1=randint(0,len(pop)-1)
        i2=randint(0,len(pop)-1)
        if (i1!=i2):
            c=self.crossover(pop[i1],pop[i2])
            c=self.mutate(c, pM)
            f1=pop[i1].fitness
            f2=pop[i2].fitness
            fc=c.fitness
            if(f1>f2) and (f1>fc):
                pop[i1]=deepcopy(c)
            if(f2>f1) and (f2>fc):
                pop[i2]=deepcopy(c)
        return pop
    
    def population(self, dimPopulation, dimIndividual):
        return [ Individual(dimIndividual) for x in range(dimPopulation) ]


    def mutate(self,individual, pM): 
        if pM > random():
            r = randint(0, individual.dimIndividual-1)
            p = randint(0, individual.dimIndividual-1)
            q = randint(0, individual.dimIndividual-1)
            z = randint(0,1)
            individual.state[r][p][z],individual.state[r][q][z] = individual.state[r][q][z],individual.state[r][p][z]
            individual.fitness = individual.fit()
        return individual
    
    def crossover(self,parent1, parent2):
        n = parent1.dimIndividual
        child=deepcopy(parent1)
        x1 = randint(0,2*n-1)
        x2 = randint(x1,2*n-1)
        for i in range(x1,x2):
            if(i < n) :
                for j in range(n):
                    child.state[i][j][0] = parent2.state[i][j][0]
            else :
                for j in range(n):
                    child.state[n-i][j][1] = parent2.state[n-i][j][1]
        child.fitness = child.fit()
        return child
    


def main():
    
    print("\n1.EA 2.Hill 3.PSO\n 4.Run program\n")
    p = int(input("N: "))
    k=0

    if (p == 4) :
        q = QApplication(sys.argv)
        widget = UserInterface()
        widget.resize(800, 600)
        widget.show()    
        q.exec()
        return
    

    listFitnessEA=[]
    listFitnessHill=[]
    listFitnessPSO=[]



    trials=1000
    indSize=4
    popSize=40
    while(k<=30):
        if p==1: 
            c=Controller()
            fitness=c.EvolutionaryAlgorithm(0,trials,indSize,popSize,random()).fitness
            listFitnessEA.append(fitness)
        elif p==2:
            c=Controller()
            fitness=c.HillClimbing(0,trials,indSize).fitness
            listFitnessHill.append(fitness)
        elif p==3:
            c=Controller()
            fitness=c.ParticleSwampOptimization(0,trials,indSize,popSize,1,1.8,1.2).fitness
            listFitnessPSO.append(fitness)
        k+=1

    if p==1:
        print("\nEvolutionary Algorithm:")
        meanEA=np.mean(listFitnessEA)
        stdEA=np.std(listFitnessEA)
        print("Mean: ")
        print(meanEA)
        print("Std Deviation: ")
        print(stdEA)
        plt.plot(listFitnessEA,'ro')

    if p==2:
        print("\nHill Climbing Algorithm:")
        meanHill=np.mean(listFitnessHill)
        stdHill=np.std(listFitnessHill)
        print("Mean: ")
        print(meanHill)
        print("Std Deviation: ")
        print(stdHill)
        plt.plot(listFitnessHill,'ro')

    if p==3:
        print("\nParticle Swamp Optimization:")
        meanPSO=np.mean(listFitnessPSO)
        stdPSO=np.std(listFitnessPSO)
        print("Mean: ")
        print(meanPSO)
        print("Std Deviation: ")
        print(stdPSO)
        plt.plot(listFitnessPSO,'ro')


main()
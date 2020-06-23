# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 09:50:36 2020

@author: dyeni
"""

from copy import copy, deepcopy

def printM(mat):
        for row in mat:
            print(row)
        print("\n")
        
        
class UserInterface:
    
    def __init__(self, N ):
        self.problem = Problem(0 , N)
        self.controller = Controller(self.problem)
                
    def run(self):
        print("1-DFS")
        print("2-Greedy")
        op = input()
        if (op == "1") : self.DFS()
        if (op == "2") : self.Greedy()
     

    def DFS(self) :
        m = self.controller.DFS()
        if m!=False:
            printM(m.m)
        else: 
            print("It failed ;(")
    
    def Greedy(self) :
        m = self.controller.Greedy()
        if m!=False:
            printM(m.m)
        else: 
            print("It failed ;(")
    
class Controller :
    
    def __init__(self, problem):
        self.problem = problem
    
    def DFS(self):
        crtState=self.problem.getRoot(self.problem.finalConfig)
        stack=[crtState]
        while len(stack)>0:
            crtState=stack.pop()
            if crtState.lastCompletedLine==self.problem.finalConfig-1:
                return crtState
            statesList=self.problem.expand(crtState,self.problem.finalConfig, crtState.lastCompletedLine)
            for state in statesList:
                stack.append(state)
        return False
    
    def Greedy(self):
        crtState=self.problem.getRoot(self.problem.finalConfig)
        for i in range(self.problem.finalConfig):
            statesList=self.problem.expand(crtState,self.problem.finalConfig, crtState.lastCompletedLine)
            if statesList==[]:
                return False
            if crtState.lastCompletedLine != -1 :
                crtState=self.problem.heuristics(statesList,crtState.lastCompletedLine,crtState.m[crtState.lastCompletedLine].index(1))
            else:
                crtState=self.problem.heuristics(statesList,-1,n)
        return crtState
    
class Problem:
    
    def __init__(self, initial, final):
        self.initialConfig = initial
        self.finalConfig = final 
        
    def expand(self, crtState, n, i):
        statesList = []
        for j in range(n):
            nextState=crtState.nextConfig(i+1,j)
            if nextState != False:
                statesList.append(nextState)
        return statesList
    
    def heuristics(self, statesList,prevRow,prevColumn):
        newStatesList=[]
        for i in range(len(statesList)):
            if statesList[i].m[0][0]==0 and statesList[i].m[0][self.finalConfig-1]==0:
                newStatesList.append([statesList[i],abs(statesList[i].m[prevRow+1].index(1)-prevColumn)])
        newStatesList.sort(key = lambda el: el[1])
        return newStatesList[0][0]
    
    def getRoot(self, N):
        initialMatrix = [0] * N
        for row in range(N):
            initialMatrix[row]= [0] * N
        state = State(initialMatrix, N, -1)
        return state
            
class State:
    
    def __init__(self, matrix, N, row):
        self.m=matrix
        self.n=N
        self.lastCompletedLine=row
        
    def nextConfig(self, i, j):
        matrix=deepcopy(self.m)
        matrix[i][j]=1
        newState=State(matrix, self.n, i)
        if newState.valid(i,j):
            return newState
        return False
    
    def valid(self,i,j):
        for k in range(self.n) : 
            if((self.m[k][j]==1 and k != i) or (self.m[i][k]==1 and k!=j)) : return False
        p,q = i-1,j-1
        while(p>=0 and q>=0) :
            if(self.m[p][q]==1):return False
            p,q=p-1,q-1
        p,q = i+1,j+1
        while(p<=self.n-1 and q<=self.n-1) :
            if(self.m[p][q]==1):return False
            p,q=p+1,q+1
        p,q = i-1,j+1
        while(p>=0 and q<=self.n-1) :
            if(self.m[p][q]==1):return False
            p,q=p-1,q+1
        p,q = i+1,j-1
        while(p<=self.n-1 and q>=0) :
            if(self.m[p][q]==1):return False
            p,q=p+1,q-1
        return True
    
        
    
n = int(input("N= "))
ui = UserInterface(n)
ui.run()
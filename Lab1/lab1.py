# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:59:50 2020

@author: dyeni
"""

import matplotlib.pyplot as plt
import numpy as np
from random import randint
from copy import deepcopy
from math import sqrt
def BinomialDistribution(trials,n,p) :
    # trials = number of trials
    # n = Number of independent experiments in each trial
    # p = Probability of success for each experiment
    s1 = np.random.binomial(n,p,trials)
    plt.plot(s1, "ro")
    plt.ylabel("bino")
    plt.axis([-0.1*trials,trials+0.1*trials,0,n+0.2*n])
    plt.show()
    
    
def UniformDistribution(trials,lowB,uppB) :
    # trials = number of trials
    # lowB = lower bound
    # uppB = upper bound
    s1 = np.random.uniform(lowB,uppB,trials)
    plt.plot(s1, "ro")
    plt.ylabel("Uniform")
    plt.axis([-0.1*trials,trials+0.1*trials,0,uppB+0.2*uppB])
    plt.show()
    
def GeometricDistribution(trials,p) :
    # trials = number of trials
    # p = probability
    s1 = np.random.geometric(p,trials)
    plt.plot(s1, "ro")
    plt.ylabel("Geometric")
    plt.axis([-0.1*trials,trials+0.1*trials,0,30])
    plt.show()
    

def main() :
    while 1>0 :
        print("Choose your poison\n")
        print("0-Binomial Distribution\n")
        print("1-Uniform Distribution\n")
        print("2-Giometrik Distribution\n")
        print("3-Sudoku\n")
        print("4-Crypt\n")
        print("5-Forms\n")
        print("6-Iesi afara\n")
        
        s = input("ior option\n-")
        if s=="6" : return
        if s=="0" : 
            t = input("Give the nr. trials\n") 
            x = input("Give the nr. of exp. in each trial\n") 
            y = input("Give the prob.\n")
            BinomialDistribution(int(t),int(x),float(y))
        if s=="1" : 
            t = input("Give the nr. trials\n")
            x = input("Give the lower bound\n") 
            y = input("Give the upper bound\n")
            UniformDistribution(int(t),int(x),int(y))
        if s=="2" : 
            t = input("Give the nr. trials\n")
            x = input("Give the probability\n") 
            GeometricDistribution(int(t), float(x))
        if s == "3" :
            sudoku()
        if s == "4" :
            crypt()
        if s == "5" :
            forms()


def goodSudokuSolution(n,matr) :
    for i in range(n):
        if(np.unique(np.array(matr[i])).size != n) : return False
    A = np.array(matr)
    for i in range(n):
        if(np.unique(A[:,i]).size != n) : return False
    i = 0
    j = 0
    m = int(sqrt(n))
    while j < m:
        if(np.unique(A[(i*m) : ((i+1)*m),(j*m) : ((j+1)*m)]).size != n) : return False
        i=i+1
        if i >= m : 
            i = 0
            j = j + 1
    return True

def printMatrix(n,matr):
    for i in range(n):
        print(matr[i])

def sudoku() :
    q = int(input("\nhow many trials?\n"))
    
    n = 4
    matr = [
        [3,0,0,2],
        [0,1,4,0],
        [1,2,0,4],
        [0,3,2,1] ]
    
    values = n*n
    k=1
    while(k <= q) :
        newMatr = deepcopy(matr)
        for i in range(n):
            for j in range(n):
                if newMatr[i][j] == 0 :
                    newValue = randint(1,n)
                    newMatr[i][j] = newValue
        if(goodSudokuSolution(n,newMatr)) : 
            print("Solution found in ",k,"trials. :D")
            printMatrix(n,newMatr)
            return
        k = k + 1
    print("A solution couldnt be found in ",q,"trials. ;(")
        
'''        
def crypt() : 
    q = int(input("how many trials?\n"))
    op = input("what operation?\n")
    n = int(input("how many words?\n"))
    words = []
    maxmLetters = 0
    for i in range(n-1):
        word=input("word "+ str(i) + ": ")[::-1]
        words.append(word)
        if len(word) > maxmLetters : maxmLetters = len(word)
    result = input("result " + ": ")[::-1]
    if len(result) > maxmLetters : maxmLetters = len(result)
    k = 1
    while(k <= q) :
        d = {}
        for word in words:
            for li in range(len(word)):
                if li == len(word)-1:
                    value = randint(1,15)
                    d[word[li]] = value
                if word[li] not in d :
                    value = randint(0,15)
           #          value=int(input(word[li] + " = "))
                    d[word[li]] = value
        
        for li in range(len(result)):
            if li == len(result)-1:
                    value = randint(1,15)
                    d[result[li]] = value
            if result[li] not in d :
                    value = randint(0,15)
           #         value=int(input(result[li] + " = "))
                    d[result[li]] = value
        d['r']=10 #scadere
        sumOfWords = []
        sums = 0
        ok = 1
        r = 0
        if op == "+" :
            for i in range(maxmLetters) :
                sums = 0
                for word in words :
                    if i < len(word) : sums += d[word[i]]
                sums += r
                r = sums // 16
                sums = sums % 16
                sumOfWords.append(sums)
            if r != 0: ok = 0
        else :
            for i in range(maxmLetters) :
                sums = d[words[0][i]]
                for j in range(1,len(words)) :
                    if i < len(words[j]) : sums -= d[words[j][i]]
                sums += r
                if(sums < 0): 
                    sums+=16
                    r = -1
                else: r = 0
                sumOfWords.append(sums)
            if r != 0: ok = 0
            while len(sumOfWords)>1 and sumOfWords[len(sumOfWords)-1]==0 : sumOfWords.pop()
        
        if len(result) != len(sumOfWords) : ok = 0
        else:
            for i in range(len(result)) :
                if d[result[i]] != sumOfWords[i] : 
                    ok = 0
                    break
        if ok == 1:
            print("\n the dict is ",d)
            print("\nsolution found in",k,"trials\n")
            return
        k+=1;
    print("\nsolution not found in ",q,"trials ;(\n")
'''        
        
    
def crypt():    
    q = int(input("how many trials?\n"))
    op = input("what operation?\n")
    n = int(input("how many words?\n"))
    words = []
    for i in range(n-1) : words.append(input("word "+ str(i) + ": ")[::-1])
    result = input("result " + ": ")[::-1]
    k = 1
    while(k <= q) :
        d = {}
        for word in words:
            for li in range(len(word)):
                if li == len(word)-1:
                    value = randint(1,15)
                    d[word[li]] = value
                if word[li] not in d :
                    value = randint(0,15)
                    d[word[li]] = value
        for li in range(len(result)):
            if li == len(result)-1:
                    value = randint(1,15)
                    d[result[li]] = value
            if result[li] not in d :
                    value = randint(0,15)
                    d[result[li]] = value
        sumz = 0
        sumr = 0
        rt = 0
        for word in words :
            p = 16
            up = 0
            sums = 0
            for let in word :
                sums+=d[let]*(p**up)
                up += 1
            if op == "-"  :
                if rt == 0 : 
                    sumz = sums
                    rt = 1
                else : sumz -= sums
            else : sumz += sums
        p = 16
        up = 0
        for let in result:
            sumr+=d[let]*(p**up)
            up += 1
        if sumr == sumz : 
            print("\n the dict is ",d)
            print("\nsolution found in",k,"trials\n")
            return
        k+=1;
    print("\nsolution not found in ",q,"trials ;(\n")
    
    
    
    
    
def forms():
    q = int(input("how many trials?\n"))
    forms=[
    [[0,0],[0,1],[0,2],[0,3]],
    [[0,0],[1,0],[1,1],[1,2],[0,2]],
    [[0,0],[1,0],[1,1],[1,2]],
    [[0,0],[0,1],[0,2],[1,2]],
    [[0,0],[1,-1],[1,0],[1,1]] ]
    
    matr = [0]*5
    for i in range(5) : matr[i] = [0]*6
    k = 1
    while k <= q:
        ok = 1
        newMatr = deepcopy(matr)
        for i in range(5):
            a = randint(0,4)
            b = randint(0,5)
            for j in range(len(forms[i])):
                if a+forms[i][j][0] >= 0 and a+forms[i][j][0] < 5 and b+forms[i][j][1]>=0 and b+forms[i][j][1]<6 and newMatr[a+forms[i][j][0]][b+forms[i][j][1]] == 0:
                    newMatr[a+forms[i][j][0]][b+forms[i][j][1]] = i
                else :
                    ok = 0
                    break
            if ok == 0 : break
        if ok == 1: 
            print("\nsolution found in",k,"trials\n")
            printMatrix(5, newMatr)
            return
        k += 1
    print("\nsolution not found in ",q,"trials ;(\n")
    
main()
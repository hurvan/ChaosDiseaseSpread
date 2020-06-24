# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:22:29 2020

@author: hurva
"""


import numpy as np
import matplotlib.pylab as plt
import copy
from mpl_toolkits.mplot3d import Axes3D



def susStep(i, t, N, sus, fullSus, inf, rem, travelMat, beta, beta1, epsilon, B, mu, my, h=0.01):
    travelSum = 0.
    for j, col in enumerate(travelMat[i]):
        if i == j:
            pass
        else:
            travelSum -= travelMat[i,j]*fullSus[i] - travelMat[j,i]*fullSus[j]
            
    newBetaEff = beta*(1+beta1*np.cos(2*np.pi*t))
    return sus + (B*N + epsilon*rem - mu*sus -newBetaEff*inf*sus/N  + travelSum )*h, travelSum


def expStep(i, t, N, exp, fullExp, sus, inf, travelMat, beta, beta1, gamma, sigma, mu, my, h=0.01):
    travelSum = 0.
    for j, col in enumerate(travelMat[i]):
        if i == j:
            pass
        else:
            travelSum -= travelMat[i,j]*fullExp[i] - travelMat[j,i]*fullExp[j]  
            
    newBetaEff = beta*(1+beta1*np.cos(2*np.pi*t))
    return exp + (newBetaEff*inf*sus/N - (sigma + mu)*exp  + travelSum )*h, travelSum


def infStep(i, exp, inf, fullInf, travelMat, beta, gamma, sigma, mu, my, h=0.01):
    travelSum = 0.
    for j, col in enumerate(travelMat[i]):
        if i == j:
            pass
        else:
            travelSum -= travelMat[i,j]*fullInf[i] - travelMat[j,i]*fullInf[j]
    return inf + (sigma*exp - (gamma + mu)*inf  + travelSum )*h, travelSum


def remStep(i, rem, fullRem, inf, travelMat, gamma, epsilon, mu,h=0.01):
    travelSum = 0.
    for j, col in enumerate(travelMat[i]):
        if i == j:
            pass
        else:
            travelSum -= travelMat[i,j]*fullRem[i] - travelMat[j,i]*fullRem[j]
    return rem + (gamma*inf - (epsilon + mu)*rem  + travelSum )*h, travelSum


def timeStep(susVec, infVec, remVec, expVec, t, travelMat, beta, beta1, gamma, epsilon, sigma, B, mu, my, lya=False, h=0.01):
    tempSus = np.copy(susVec)
    tempInf = np.copy(infVec)
    tempRem = np.copy(remVec)
    tempExp = np.copy(expVec)

    for i, row in enumerate(susVec):  
        N = susVec[i] + expVec[i] + remVec[i] + expVec[i]
        tempSus[i], ts1 = susStep(i, t, N, susVec[i], susVec, infVec[i], remVec[i], travelMat, beta, beta1, epsilon, B, mu, my, h)
        tempInf[i], ts2 = infStep(i, expVec[i], infVec[i], infVec, travelMat, beta, gamma, sigma, mu, my, h)
        tempRem[i], ts3 = remStep(i, remVec[i], remVec, infVec[i], travelMat, gamma, epsilon, mu, h)
        tempExp[i], ts4 = expStep(i, t, N, expVec[i], expVec, susVec[i], infVec[i], travelMat, beta, beta1, gamma, sigma, mu, my, h)
                
        # if lya:
        #     tSum = 0
        #     # for j ,row in enumerate(travelMat[i]):
        #     #     if i == j:
        #     #         pass
        #     #     else:
        #     #         tSum += travelMat[i,j]
                
            
        #     jacobian = np.array([[-beta/N*tempInf[i] - tSum, 0, -beta/N*tempSus[i], epsilon],
        #                         [beta/N*tempInf[i], -sigma-tSum, beta/N*tempSus[i], 0],
        #                         [0, sigma, -gamma-tSum, 0],
        #                         [0, 0, gamma, -epsilon-tSum]])
            
        #     #tr = np.trace(jacobian)
        #     eig = np.linalg.eig(jacobian)
    
    if lya:
        return tempSus, tempInf, tempRem, tempExp, t, eig   
    else:
        return tempSus, tempInf, tempRem, tempExp, t, np.array([0,0,0,0])



gamma = 100
beta = 1800
beta1 = 0.28
epsilon = 0.00
sigma = 35.48
B = 0.02
mu = 0.02
my = 0
h = 0.001

totalSteps = 50000


populationList = np.array([159606, 287966, 59686, 287382, 333848, 130810,
                           363599, 245446, 201469, 250093, 1377827, 2377081, 
                           297540, 383713, 282414,271736, 245347, 275845, 
                           1725881, 304805, 465495])

populationList = populationList/np.max(populationList)

xSize = len(populationList)

travelMat = np.zeros((xSize,xSize))



northToSouth = [20,6,16,5,19,3,14,18,17,1,21,10,12,7,8,2,4,9,15,11,13]


neighbors = [[21,17,18],
            [3,5,9,11,8],
            [18,10],
            [4,3,6,7],
            [21,15,14,17],
            [6,5,4,2],
            [15,13,18,17,19],
            [16,13,14,17,20],
            [20,21,19,14,18],
            [2],
            [19,20,17],
            [7,9,12,16],
            [10,7,9,11,13],
            [5,9,12,10],
            [6,11,15],
            [1,4,3],
            [2,3,5],
            [7,6,11,12,10],
            [8,11,13,14,19],
            [6,9,12,13,15,8],
            [12,11,15,14,18]]


        
for i, row in enumerate(travelMat): 
    for j, col in enumerate(row):     
        for nr in neighbors[i]:
            if nr == northToSouth[j]:
                # travelMat[i,j] = populationList[i]+populationList[j]
                travelMat[i,j] = 1.        
             
travelMat *= 0.0000001


plt.ion()


###
# Outer loop for different initial conditions
###

for j in range(2):
    susVec = populationList
    #susVec = np.ones((xSize))
    infVec = np.zeros((xSize))
    remVec = np.zeros((xSize))
    expVec = np.zeros((xSize))
    
    infVec[11] = susVec[11]*(j+1)/200000.  # different initial conditions
    susVec[11] -= susVec[11]*(j+1)/200000. # different initial conditions

    
    susPlot = []
    infPlot = []
    remPlot = []
    expPlot = []
    timePlot = []
            
    print(j)
    t = 0
    for i in range(totalSteps):
        susVec, infVec, remVec, expVec, t, eig = timeStep(susVec, infVec, remVec, expVec, t, travelMat, beta, beta1, gamma, epsilon, sigma, B, mu, my,False, h)
        t += h
        susPlot.append(susVec)
        infPlot.append(infVec)
        remPlot.append(remVec)
        expPlot.append(expVec)

        timePlot.append(t)
        
        
        if i%10000==0:
            print(i/totalSteps)
        
        # if i%5000 == 0 and i > 5000:
        #     plt.cla()
        #     plt.plot(timePlot,np.sum(susPlot,axis=1),'g')
        #     plt.plot(timePlot,np.sum(infPlot,axis=1),'r')
        #     plt.plot(timePlot,np.sum(remPlot,axis=1),'b')
        #     plt.plot(timePlot,np.sum(expPlot,axis=1),'k')
        #     #plt.plot((np.diff(np.sum(np.array(susPlot),axis=1))/h)[1000::], (np.diff(np.sum(np.array(infPlot),axis=1))/h)[1000::],'r')
        #     #plt.plot(np.sum(np.array(susPlot),axis=1)[1000::], np.sum(np.array(infPlot),axis=1)[1000::])
        # #     # print(infVec[1])
            
        # #     print(np.sum(susPlot[-1]+infPlot[-1]+remPlot[-1]+expPlot[-1]))
        
        # #     # plt.imshow(np.array([infVec]).reshape(-1,21),aspect='auto',interpolation='none',vmin=0,vmax=1)
        #     plt.pause(0.001)
        #     plt.draw()
        
    


    ############
    # For plotting the phase space?  (susceptible rate VS infected rate)
    ############
    
    # plt.plot((np.diff(np.sum(np.array(susPlot),axis=1))/h)[500::], (np.diff(np.sum(np.array(infPlot),axis=1))/h)[500::])

    ############
    # For plotting the phase space? (susceptible VS infected)
    ############

    plt.plot(np.sum(susPlot,axis=1)[500::],np.sum(infPlot,axis=1)[500::])

    ############
    # For plotting sum of all regions 
    ############
    
    
    # plt.plot(timePlot[50000::],np.sum(susPlot,axis=1)[50000::],'g')
    # plt.plot(timePlot[50000::],np.sum(infPlot,axis=1)[50000::],'r')
    # plt.plot(timePlot[50000::],np.sum(remPlot,axis=1)[50000::],'b')
    # plt.plot(timePlot[50000::],np.sum(expPlot,axis=1)[50000::],'k')
    
    # plt.plot(timePlot[0::],np.sum(susPlot,axis=1)[0::],'g')
    # plt.plot(timePlot[0::],np.sum(infPlot,axis=1)[0::],'r')
    # plt.plot(timePlot[0::],np.sum(remPlot,axis=1)[0::],'b')
    # plt.plot(timePlot[0::],np.sum(expPlot,axis=1)[0::],'k')
    

    plt.pause(0.01)
    plt.draw()

    
    ############
    # For plotting all regions seperatly
    ############
    
    # susPlot = np.array(susPlot).T
    # infPlot = np.array(infPlot).T
    # remPlot = np.array(remPlot).T
    # expPlot = np.array(expPlot).T
    
    # for i in range(xSize):
        # plt.plot(timePlot,susPlot[i],'g')
        # plt.plot(timePlot,infPlot[i],'r')
        # plt.plot(timePlot,remPlot[i],'b')
        # plt.plot(timePlot,expPlot[i],'k')
    




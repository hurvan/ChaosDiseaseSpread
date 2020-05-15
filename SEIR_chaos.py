# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:43:45 2020

@author: hurva
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pylab as plt
import copy



########
## Need to change the name of all the constant for the sake of concistency 
## because I have not followed the convention. Example beta is usually infectivity 
## but i used the name alpha. 


####
## The alphaEff is the effective infectivity because of seasonality. alpha is 
## the baserate, alpha1 is the strength of the "seasonality"
##

def susStep(N, t, sus, inf, rem, alpha, alpha1, gamma, B, d, h=0.01):
    alphaEff = alpha*(1+alpha1*np.cos(np.pi*2*t))
    return sus + (B*N + gamma*rem - d*sus -alphaEff*inf*sus/N)*h

def expStep(N, t, exp, sus, inf, alpha, alpha1, beta, delta, d, h=0.01):
    alphaEff = alpha*(1+alpha1*np.cos(np.pi*2*t))
    return exp + (alphaEff*inf*sus/N - (delta + d)*exp)*h


def infStep(exp, inf, alpha, beta, delta, d, h=0.01):
    return inf + (delta*exp - (beta + d)*inf)*h

def remStep(rem, inf, beta, gamma, d,h=0.01):
    return rem + (beta*inf - (gamma + d)*rem)*h






def timeStep(susMat, infMat, remMat, expMat, t, alpha, alpha1, beta, gamma, delta, B, d, h=0.01):
    tempSus = np.copy(susMat)
    tempInf = np.copy(infMat)
    tempRem = np.copy(remMat)
    tempExp = np.copy(expMat)

    for i, row in enumerate(susMat):     
        for j, col in enumerate(row):
            N = susMat[i,j] + remMat[i,j] + infMat[i,j] + expMat[i,j]
            tempSus[i,j] = susStep(N, t, susMat[i,j], infMat[i,j], remMat[i,j], alpha, alpha1, gamma, B, d, h)
            tempInf[i,j] = infStep(expMat[i,j], infMat[i,j], alpha, beta, delta, d, h)
            tempRem[i,j] = remStep(remMat[i,j], infMat[i,j], beta, gamma, d, h)
            tempExp[i,j] = expStep(N, t, expMat[i,j], susMat[i,j], infMat[i,j], alpha, alpha1, beta, delta, d, h)

    return tempSus, tempInf, tempRem, tempExp, t                  

    





xSize = 1
ySize = 1
beta = 100
alpha = 1800          
alpha1 = 0.28
gamma = 0.00
delta = 35.48
B = 0.02
d = 0.02
h = 0.001


plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


###
# Plot 5 different starting conditions
#

for j in range(5):
    susMat = np.ones((xSize,ySize))  
    infMat = np.zeros((xSize,ySize))
    remMat = np.zeros((xSize,ySize))
    expMat = np.zeros((xSize,ySize))
    
    infMat[0,0] = susMat[0,0]*(j+1)/100
    susMat[0,0] -= susMat[0,0]*(j+1)/100
    
    t = 0
    
    susPlot = []
    infPlot = []
    remPlot = []
    expPlot = []
    timePlot = []
        
    print(j)
    
    ###
    # 50000 iterations seems enough for the system to behave somewhat asymptotically
    #
    
    for i in range(50000):
        susMat, infMat, remMat, expMat, t = timeStep(susMat, infMat, remMat, expMat, t, alpha, alpha1, beta, gamma, delta, B, d, h)
        t += h
        susPlot.append(np.sum(np.sum(susMat)))
        infPlot.append(np.sum(np.sum(infMat)))
        remPlot.append(np.sum(np.sum(remMat)))
        expPlot.append(np.sum(np.sum(expMat)))
        
        timePlot.append(t)
        
        
        # if i%100 == 0 and i >500:
        #     plt.cla()
        #     plt.plot((np.diff(np.array(susPlot))/h)[500::],(np.diff(np.array(infPlot))/h)[500::])
        #     plt.plot((np.diff(np.array(susPlot))/h)[-1],(np.diff(np.array(infPlot))/h)[-1],'r*',markersize=10)
        #     plt.plot(susPlot,'g')
        #     plt.plot(infPlot,'r')
        #     plt.plot(remPlot,'b')
        #     plt.plot(expPlot,'k')
            
        #     print(susPlot[-1]+infPlot[-1]+remPlot[-1]+expPlot[-1])
        
        #     # plt.imshow(infMat,aspect='auto',interpolation='none',vmin=0,vmax=1/(xSize*ySize))
            # plt.pause(0.001)
            # plt.draw()
        

    #ax.plot((np.diff(np.array(susPlot))/h)[500::],(np.diff(np.array(remPlot))/h)[500::],(np.diff(np.array(infPlot))/h)[500::])
    # ax.pause(0.001)
    # ax.draw()
    #plt.plot(np.diff(np.array(susPlot)),np.diff(np.array(infPlot)+np.array(expPlot)))
    
    plt.plot(timePlot,susPlot,'g')
    plt.plot(timePlot,infPlot,'r')
    plt.plot(timePlot,remPlot,'b')
    plt.plot(timePlot,expPlot,'k')
plt.show()




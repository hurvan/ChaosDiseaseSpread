# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:51:49 2020

@author: hurva
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pylab as plt
import cv2
import copy

periodic = False


if periodic:
    bType = cv2.BORDER_WRAP
else:
    bType = cv2.BORDER_REFLECT


def Diffuse(matrix,D,h=0.01):
    temp = np.copy(matrix)
    border=cv2.copyMakeBorder(matrix, top=1, bottom=1, left=1, right=1, borderType=bType)
    laplacian = cv2.Laplacian(border,cv2.CV_64F,ksize=1)
    laplacian = laplacian[1:-1,1:-1]
    for i, row in enumerate(matrix):
        for j, col in enumerate(row):
            temp[i,j] += (D*laplacian[i,j])*h    
    return temp


def susStep(sus, inf, rem, alpha, gamma, B, d, my, h=0.01):
    return sus + (B + gamma*rem - d*sus -alpha*inf*(1-my*inf)*sus)*h

def expStep(exp, sus, inf, alpha, beta, delta, d, my, h=0.01):
    return exp + (alpha*inf*(1-my*inf)*sus - (delta + d)*exp)*h


def infStep(exp, inf, alpha, beta, delta, d, my, h=0.01):
    return inf + (delta*exp - (beta + d)*inf)*h

def remStep(rem, inf, beta, gamma, d,h=0.01):
    return rem + (beta*inf - (gamma + d)*rem)*h






def timeStep(susMat, infMat, remMat, expMat, alpha, beta, gamma, delta, B, d, my, Ds, Di, Dr, h=0.01):
    tempSus = np.copy(susMat)
    tempInf = np.copy(infMat)
    tempRem = np.copy(remMat)
    tempExp = np.copy(expMat)

    for i, row in enumerate(susMat):     
        for j, col in enumerate(row):
            
            tempSus[i,j] = susStep(susMat[i,j], infMat[i,j], remMat[i,j], alpha, gamma, B, d, my, h)
            tempInf[i,j] = infStep(expMat[i,j], infMat[i,j], alpha, beta, delta, d, my, h)
            tempRem[i,j] = remStep(remMat[i,j], infMat[i,j], beta, gamma, d, h)
            tempExp[i,j] = expStep(expMat[i,j], susMat[i,j], infMat[i,j], alpha, beta, delta, d, my, h)
            
            if Ds+Di+Dr != 0:
                tempSus = Diffuse(tempSus, Ds, h)
                tempInf = Diffuse(tempInf, Di, h)
                tempRem = Diffuse(tempRem, Dr, h)
                tempExp = Diffuse(tempExp, Ds, h)

    return tempSus, tempInf, tempRem, tempExp                     




xSize = 1
ySize = 1
beta = 0.1           #1/21
alpha = 0.2
gamma = 0.001
delta = 0.1
B = 0.0001
d = 0.0001
my = 0.5
Ds = 0.0000000000
Di = 0.0000000000
Dr = 0.0000000000
h = 0.5




susMat = np.ones((xSize,ySize))  
# susMat = np.array([[0.1,0.5,1.0],
#                    [0.1,0.3,0.7],
#                    [0.3,0.2,0.3]])
infMat = np.zeros((xSize,ySize))
remMat = np.zeros((xSize,ySize))
expMat = np.zeros((xSize,ySize))

infMat[0,0] = susMat[0,0]*0.01
susMat[0,0] -= susMat[0,0]*0.01


susPlot = []
infPlot = []
remPlot = []
expPlot = []

#plt.ion()

for j in range(5):
    susMat = np.ones((xSize,ySize))  
    infMat = np.zeros((xSize,ySize))
    remMat = np.zeros((xSize,ySize))
    expMat = np.zeros((xSize,ySize))
    
    infMat[0,0] = susMat[0,0]*0.2*(j+1)
    susMat[0,0] -= susMat[0,0]*0.2*(j+1)
    
    
    susPlot = []
    infPlot = []
    remPlot = []
    expPlot = []
        
    print(j)
    for i in range(10000):
        susMat, infMat, remMat, expMat = timeStep(susMat, infMat, remMat, expMat, alpha, beta, gamma, delta, B, d, my, Ds, Di, Dr, h)
    
        susPlot.append(np.sum(np.sum(susMat)))
        infPlot.append(np.sum(np.sum(infMat)))
        remPlot.append(np.sum(np.sum(remMat)))
        expPlot.append(np.sum(np.sum(expMat)))
              
        # if i%100 == 0:
        #     plt.cla()
        #     plt.plot(susPlot,'g')
        #     plt.plot(infPlot,'r')
        #     plt.plot(remPlot,'b')
        #     plt.plot(expPlot,'k')
            
        #     print(susPlot[-1]+infPlot[-1]+remPlot[-1]+expPlot[-1])
        
        #     # plt.imshow(infMat,aspect='auto',interpolation='none',vmin=0,vmax=1/(xSize*ySize))
        #     plt.pause(0.001)
        #     plt.draw()

    #plt.plot(np.diff(np.array(susPlot)),np.diff(np.array(infPlot)+np.array(expPlot))) #Plots the phase diagram
    
    plt.plot(susPlot,'g')
    plt.plot(infPlot,'r')
    plt.plot(remPlot,'b')
    plt.plot(expPlot,'k')
plt.show()



# sp = np.fft.fft(susPlot)
# freq = np.fft.fftfreq(len(susPlot))
# plt.semilogy(freq, np.abs(sp))
# plt.show()




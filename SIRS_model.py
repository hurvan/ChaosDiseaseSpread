# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:04:38 2020

@author: hurva
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:50:36 2020

@author: hurva
"""

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

def infStep(sus, inf, alpha, beta, d, my, h=0.01):
    return inf + (alpha*inf*(1-my*inf)*sus - (beta + d)*inf)*h

def remStep(rem, inf, beta, gamma, d,h=0.01):
    return rem + (beta*inf - (gamma + d)*rem)*h





def timeStep(susMat, infMat, remMat, alpha, beta, gamma, B, d, my, Ds, Di, Dr, h=0.01):
    tempSus = np.copy(susMat)
    tempInf = np.copy(infMat)
    tempRem = np.copy(remMat)

    for i, row in enumerate(susMat):     
        for j, col in enumerate(row):
            
            tempSus[i,j] = susStep(susMat[i,j], infMat[i,j], remMat[i,j], alpha, gamma, B, d, my, h)
            tempInf[i,j] = infStep(susMat[i,j], infMat[i,j], alpha, beta, d, my, h)
            tempRem[i,j] = remStep(remMat[i,j], infMat[i,j], beta, gamma, d, h)
            
            if Ds+Di+Dr != 0:
                tempSus = Diffuse(tempSus, Ds, h)
                tempInf = Diffuse(tempInf, Di, h)
                tempRem = Diffuse(tempRem, Dr, h)

    return tempSus, tempInf, tempRem                        




xSize = 1
ySize = 1
beta = 0.1           #1/21
alpha = 0.2  #2.5 / beta
gamma = 0.001
B = 0.00
d = 0.00
my = 0.5
Ds = 0.000
Di = 0.000
Dr = 0.000
h = 0.05

susMat = np.ones((xSize,ySize)) / (xSize*ySize) 
infMat = np.zeros((xSize,ySize))
remMat = np.zeros((xSize,ySize))
wanMat = np.zeros((xSize,ySize))+0.0000001

infMat[0,0] = susMat[0,0]*0.001
susMat[0,0] -= susMat[0,0]*0.001



susPlot = []
infPlot = []
remPlot = []
wanPlot = []

plt.ion()

for i in range(100000):
    susMat, infMat, remMat = timeStep(susMat, infMat, remMat, alpha, beta, gamma, B, d, my, Ds, Di, Dr, h)

    susPlot.append(np.sum(np.sum(susMat)))
    infPlot.append(np.sum(np.sum(infMat)))
    remPlot.append(np.sum(np.sum(remMat)))
    
    if i%100 == 0:
        plt.cla()
        plt.plot(susPlot,'g')
        plt.plot(infPlot,'r')
        plt.plot(remPlot,'b')
        print(susPlot[-1]+infPlot[-1]+remPlot[-1])
        plt.pause(0.001)
        plt.draw()
    
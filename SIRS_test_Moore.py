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
import matplotlib.animation as animation
from  matplotlib.animation import FuncAnimation
import cv2
import copy

periodic = False


if periodic:
    bType = cv2.BORDER_WRAP
else:
    bType = cv2.BORDER_REFLECT


def Diffuse(inf, sus,D,h=0.01):
    tempInf = np.zeros(((xSize + 2), (ySize + 2)))
    tempInf[1:(xSize+1), 1:(ySize+1)] = np.copy(inf)

    tempSus = np.zeros(((xSize + 2), (ySize + 2)))
    tempSus[1:(xSize+1), 1:(ySize+1)] = np.copy(sus)

    for i, row in enumerate(inf):
        i += 1
        for j, col in enumerate(row):
            j += 1
            neighb = tempInf[(i-1):(i+1), (j-1):(j+1)]
            diff = (np.sum(neighb*D) - tempInf[i, j]*D)
            tempInf[i, j] += diff 
            tempSus[i, j] += -diff

    retInf = np.copy(tempInf[1:(xSize+1), 1:(ySize+1)])
    retSus = np.copy(tempSus[1:(xSize+1), 1:(ySize+1)])

    print()
    print(retInf)
    print(tempInf)
    print(retSus)
    print(tempInf)

    return retInf, retSus

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

    tempSus = susStep(susMat, infMat, remMat, alpha, gamma, B, d, my, h)
    tempInf = infStep(susMat, infMat, alpha, beta, d, my, h)
    tempRem = remStep(remMat, infMat, beta, gamma, d, h)

            
    if Ds+Di+Dr != 0:
        tempInf, tempSus, = Diffuse(tempInf, tempSus, Ds, h)

    return tempSus, tempInf, tempRem                        


option = 3

xSize = 3
ySize = 3
iters = 1000
beta = 0.1           #1/21
alpha = 0.2  #2.5 / beta
gamma = 0.001
B = 0.00
d = 0.00
my = 0.5
Ds = 0.005
Di = 0.005
Dr = 0.005
h = 0.05

susMat = np.ones((xSize, ySize, iters))
infMat = np.zeros((xSize, ySize, iters))
remMat = np.zeros((xSize, ySize, iters))
wanMat = np.zeros((xSize, ySize, iters))+0.0000001

infMat[0,0,0] = susMat[0,0,0]*0.001
susMat[0,0,0] -= susMat[0,0,0]*0.001

for i in range(iters-1):
    tempSus, tempInf, tempRem = timeStep(susMat[:,:,i], infMat[:,:,i],
            remMat[:,:,i], alpha, beta, gamma, B, d, my, Ds, Di, Dr, h)

    susMat[:,:,(i+1)] = tempSus 
    infMat[:,:,(i+1)] = tempInf
    remMat[:,:,(i+1)] = tempRem

if option == 0:
    plt.plot(susMat[0,0,:],'g')
    plt.plot(infMat[0,0,:],'r')
    plt.plot(remMat[0,0,:],'b')
    plt.show()
if option == 1:
    for n in range(xSize):
        for m in range (ySize):
            plt.plot(susMat[n, m, :])
    plt.show()
if option == 2:
    plt.matshow(susMat[:, :, -1])
    plt.show()

if option == 3:
    fig = plt.figure()

    pltMat = infMat
    def f(time):
        return pltMat[:, :, time]

    time = -1;

    im = plt.imshow(f(time), animated=True, vmin=np.min(pltMat),
            vmax=np.max(pltMat))

    def updatefig(*args):
        global time
        time += 1
        if time > iters:
            time = 0
        print(time)
        v = f(time)
        print(v)
        im.set_array(f(time))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, frames=10, interval=0.1, blit=True)
    plt.show()       

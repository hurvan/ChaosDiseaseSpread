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

global verbal
verbal = 4

if periodic:
    bType = cv2.BORDER_WRAP
else:
    bType = cv2.BORDER_REFLECT


def Diffuse(inf, h=0.01):
    tempInf = np.zeros(((xSize + 2), (ySize + 2)))
    tempInf[1:(xSize+1), 1:(ySize+1)] = np.copy(inf)

    retInf = np.zeros((xSize, ySize))

    if verbal == 1 or verbal == 1.1:
        print("tempInf: \n", tempInf)

    for i, row in enumerate(inf):
        i += 1

        for j, col in enumerate(row):
            j += 1

            neighb = np.copy(tempInf[(i-1):(i+2), (j-1):(j+2)])
            neighb[1, 1] = 0
            diff = (np.sum(neighb))
            if verbal == 1:
                print("diff: \n", diff)
            #if(tempInf[i, j] + diff) < 0:
            #    diff = 0

            retInf[(i-1), (j-1)] = diff

    if verbal == 1 or verbal == 1.1:
        print()
        print("retInf: \n", retInf)
        print("tempInf: \n", tempInf)

    return retInf

def susStep(sus, inf, rem, alpha, gamma, B, d, my, diff, h=0.01):
    return sus + (B + gamma*rem - d*sus -alpha*inf*(1-my*inf)*sus - (diff * sus))*h

def infStep(sus, inf, alpha, beta, d, my, diff, h=0.01):
    return inf + (alpha*inf*(1-my*inf)*sus - (beta + d)*inf + (diff * sus))*h

def remStep(rem, inf, beta, gamma, d, h=0.01):
    return rem + (beta*inf - (gamma + d)*rem)*h





def timeStep(susMat, infMat, remMat, alpha, beta, gamma, B, d, my, Ds, Di, Dr, h=0.01):
    tempSus = np.copy(susMat)
    tempInf = np.copy(infMat)
    tempRem = np.copy(remMat)

    diff = Ds * Diffuse(tempInf)

    tempSus = susStep(susMat, infMat, remMat, alpha, gamma, B, d, my, diff, h)
    tempInf = infStep(susMat, infMat, alpha, beta, d, my, diff, h)
    tempRem = remStep(remMat, infMat, beta, gamma, d, h)

    return tempSus, tempInf, tempRem


option = 1

xSize = 3
ySize = 3
iters = 10000
beta = 0.1           #1/21
alpha = 0.2  #2.5 / beta
gamma = 0.001
B = 0.00
d = 0.00
my = 0.5
Ds = 0.05
Di = 0.0
Dr = 0.0
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
            c1 = [((n + m)/(np.max(xSize) + np.max(ySize))), 0, 0]
            c2 = [0, ((n + m)/(np.max(xSize) + np.max(ySize))), 0]
            c3 = [0, 0, ((n + m)/(np.max(xSize) + np.max(ySize)))]

            plt.plot(susMat[n, m, :], color = c1)
            plt.plot(infMat[n, m, :], color = c2)
            plt.plot(remMat[n, m, :], color = c3)

    plt.legend(["Susceptible", "Infected", "Removed"])
    plt.show()
    if verbal == 4:
        print("Sus: \n", susMat[:, :, -1])
        print("Inf: \n", infMat[:, :, -1])
        print("Rem: \n", remMat[:, :, -1])

if option == 2:
    plt.matshow(susMat[:, :, -1])
    plt.show()

if option == 3:
    fig = plt.figure()

    def f1(time):
        return susMat[:, :, time]

    def f2(time):
        return infMat[:, :, time]

    time = -1;

    plt.subplot(1, 2, 1)
    im1 = plt.imshow(f1(time), animated=True, vmin=np.min(susMat), vmax=np.max(susMat))
    plt.title('Susceptibale')

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(f2(time), animated=True, vmin=np.min(infMat), vmax=np.max(infMat))
    plt.title('Infected')

    def updatefig(*args):
        global time
        time += 1
        if time >= iters:
            time = 0
        print(time)
        v = f1(time)
        print(v)
        im1.set_array(f1(time))
        im2.set_array(f2(time))
        return im1, im2,

    ani = animation.FuncAnimation(fig, updatefig, frames=10, interval=0.1, blit=True)
    plt.show()

if option == 4:
    fig = plt.figure()

    def f1(time):
        return susMat[:, :, time]

    def f2(time):
        return infMat[:, :, time]

    def f3(time):
        return remMat[:, :, time]

    def iniPlot():
        pS = np.empty((xSize, ySize), dtype=object)
        pI = np.empty((xSize, ySize), dtype=object)
        pR = np.empty((xSize, ySize), dtype=object)
        for n in range(xSize):
            for m in range (ySize):
                c1 = [((n + m)/(np.max(xSize) + np.max(ySize))), 0, 0]
                c2 = [0, ((n + m)/(np.max(xSize) + np.max(ySize))), 0]
                c3 = [0, 0, ((n + m)/(np.max(xSize) + np.max(ySize)))]

                pS[n, m] = plt.plot([np.nan] * (iters - 1), color = c1)
                pI[n, m] = plt.plot([np.nan] * (iters - 1), color = c2)
                pR[n, m] = plt.plot([np.nan] * (iters - 1), color = c3)

        plt.legend(["Susceptible", "Infected", "Removed"])
        return pS, pI, pR

    def updPlot(time):
        for n in range(xSize):
            for m in range (ySize):
                ydataS = susMat[n, m, 0:time]
                pS[n, m][0].set_ydata()
                pI[n, m][0].set_ydata(infMat[n, m, time])
                pR[n, m][0].set_ydata(remMat[n, m, time])
        return pS, pI, pR

    time = -1;

    plt.subplot(2, 2, 1)
    im1 = plt.imshow(f1(time), animated=True, vmin=np.min(susMat), vmax=np.max(susMat))
    plt.title('Susceptibale')

    plt.subplot(2, 2, 2)
    im2 = plt.imshow(f2(time), animated=True, vmin=np.min(infMat), vmax=np.max(infMat))
    plt.title('Infected')

    plt.subplot(2, 2, 3)
    im3 = plt.imshow(f2(time), animated=True, vmin=np.min(remMat), vmax=np.max(remMat))
    plt.title('Infected')

    plt.subplot(2, 2, 4)
    pS, pI, pR = iniPlot()

    def updatefig(*args):
        global time
        time += 1
        if time >= iters:
            time = 0
        print(time)
        v = f1(time)
        print(v)
        im1.set_array(f1(time))
        im2.set_array(f2(time))
        im3.set_array(f3(time))
        updPlot(time)
        return im1, im2, im3, pS[0, 0][0],

    ani = animation.FuncAnimation(fig, updatefig, frames=10, interval=0.1, blit=True)
    plt.show()

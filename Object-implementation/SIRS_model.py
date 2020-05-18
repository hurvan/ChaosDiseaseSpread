import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from  matplotlib.animation import FuncAnimation
from scipy import signal
import cv2
import copy


def SIRS_model(iters, xSize, ySize, beta, gamma, lambd, my, delta, m, N):
    def conv(inf, h=0.01):
        retInf = np.copy(inf)
        retInf = signal.convolve2d(retInf, np.ones((3, 3)), boundary='symm', mode='same')

        return retInf
        
    def susStep(S, I, R, beta, lambd, my, delta, N, diff, h):
        return S + ((lambd*S) + (delta*R) - (my*S) - (beta*S*I)/N - (diff*S)/N)*h

    def infStep(S, I, beta, gamma, my, N, diff, h):
        return I + ((beta*S*I)/N + (diff*S)/N - gamma*I - my*I)*h

    def remStep(I, R, gamma, my, delta, h):
        return R + (gamma*I - my*R - delta*R)*h

    def timeStep(susMat, infMat, remMat, beta, gamma, lambd, my, delta, N, m, h):
        tempSus = np.copy(susMat)
        tempInf = np.copy(infMat)
        tempRem = np.copy(remMat)

        diff = m * conv(tempInf)

        tempSus = susStep(susMat, infMat, remMat, beta, lambd, my, delta, N, diff, h)
        tempInf = infStep(susMat, infMat, beta, gamma, my, N, diff, h)
        tempRem = remStep(infMat, remMat, gamma, my, delta, h)
        
        return tempSus, tempInf, tempRem

    ## -------- Set up -------- ##
    iters = iters
    xSize = xSize
    ySize = ySize
    beta = beta
    gamma = gamma
    lambd = lambd
    my = my
    delta = delta
    m = m
    N = N
    h = 0.05

    susMat = np.zeros((xSize, ySize, iters))
    infMat = np.zeros((xSize, ySize, iters))
    remMat = np.zeros((xSize, ySize, iters))
    wanMat = np.zeros((xSize, ySize, iters))+0.0000001

    susMat[:, :, 0] = np.ones((xSize, ySize)) * N 
    infMat[0,0,0] = susMat[0,0,0]*0.001
    infMat[3,3,0] = susMat[0,0,0]*0.002
    susMat[:,:,0] -= infMat[:,:,0]

    print("Sus: \n", susMat[:, :, 0])
    print("Inf: \n", infMat[:, :, 0])
    print("Rem: \n", remMat[:, :, 0])
    print()

    ## -------- Run simulation -------- ##
    for i in range(iters-1):
        tempSus, tempInf, tempRem = timeStep(susMat[:,:,i], infMat[:,:,i], remMat[:,:,i], beta, gamma, lambd, my, delta, N, m, h)

        susMat[:,:,(i+1)] = tempSus
        infMat[:,:,(i+1)] = tempInf
        remMat[:,:,(i+1)] = tempRem

        N = tempSus + tempInf + tempRem

    return susMat, infMat, remMat



import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from  matplotlib.animation import FuncAnimation
from scipy import signal
import cv2
import copy


def SIRSs_model(iters, xSize, ySize, beta, gamma, lambd, my, delta, omega, m, N, inf):
    #Convolution for spatial spread. Change np.ones to change window
    def conv(inf, h=0.01):
        retInf = np.copy(inf)
        retInf = signal.convolve2d(retInf, np.ones((3, 3)), boundary='symm', mode='same')

        return retInf

    #Time step for dS/dt 
    def susStep(S, I, R, beta, lambd, my, delta, omega, t, N, diff, h):
        return S + ((lambd*S) + (delta*R) - (my*S) - (((1 + omega*np.cos((2*np.pi*365)*t))*beta)*S*I)/N - (diff*S)/N)*h

    #Time step for dI/dt 
    def infStep(S, I, beta, gamma, my, omega, t, N, diff, h):
        return I + ((((1 + omega*np.cos((2*np.pi*365)*t))*beta)*S*I)/N + (diff*S)/N - gamma*I - my*I)*h

    #Time step for dR/dt 
    def remStep(I, R, gamma, my, delta, h):
        return R + (gamma*I - my*R - delta*R)*h

    #Time step for Euler-method
    def timeStep(susMat, infMat, remMat, beta, gamma, lambd, my, delta, omega, t, N, m, h):
        tempSus = np.copy(susMat)
        tempInf = np.copy(infMat)
        tempRem = np.copy(remMat)

        if m != 0:
            diff = m * conv(tempInf)
        else:
            diff = 0

        tempSus = susStep(susMat, infMat, remMat, beta, lambd, my, delta, omega, t, N, diff, h)
        tempInf = infStep(susMat, infMat, beta, gamma, my, omega, t, N, diff, h)
        tempRem = remStep(infMat, remMat, gamma, my, delta, h)

        return tempSus, tempInf, tempRem

    ## -------- Set up -------- ##
    h = 0.05

    susMat = np.zeros((xSize, ySize, iters))
    infMat = np.zeros((xSize, ySize, iters))
    remMat = np.zeros((xSize, ySize, iters))
    wanMat = np.zeros((xSize, ySize, iters))+0.0000001

    susMat[:, :, 0] = np.ones((xSize, ySize)) * N 
    infMat[:,:,0] = susMat[:,:,0]*inf
    susMat[:,:,0] -= infMat[:,:,0]

    print("Sus: \n", susMat[:, :, 0])
    print("Inf: \n", infMat[:, :, 0])
    print("Rem: \n", remMat[:, :, 0])
    print()

    ## -------- Run simulation -------- ##
    t = 0
    for i in range(iters-1):
        tempSus, tempInf, tempRem = timeStep(susMat[:,:,i], infMat[:,:,i], remMat[:,:,i], beta, gamma, lambd, my, delta, omega, t, N, m, h)
        t += h

        susMat[:,:,(i+1)] = tempSus
        infMat[:,:,(i+1)] = tempInf
        remMat[:,:,(i+1)] = tempRem

        N = tempSus + tempInf + tempRem

    return susMat, infMat, remMat



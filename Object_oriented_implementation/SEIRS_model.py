import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from  matplotlib.animation import FuncAnimation
from scipy import signal
import cv2
import copy


def SEIRSs_model(iters, xSize, ySize, alpha, beta, gamma, lambd, my, delta, omega, m, N, inf, h=0.01):

    verbal = 1

    #Convolution for spatial spread. 
    def conv(inf, h=0.01):
        retInf = np.copy(inf)
        window = np.ones((3, 3))
        window[1, 1] = 0
        retInf = signal.convolve2d(retInf, window, boundary='symm', mode='same')

        return retInf

    #Time step for dS/dt 
    def susStep(S, I, R, beta, lambd, my, delta, omega, t, N, diff, h):
        #return S + ((lambd*N) + (delta*R) - (my*S) - (((1 + omega*np.cos((2*np.pi*t*365)))*beta)*S*I)/N - (diff*S)/N)*h
        betaT = (1 + omega*np.cos(2*np.pi*t))*beta
        return S + ((lambd*N) + (delta*R) - (my*S) - (betaT*S*I)/N - (diff*S)/N)*h

    #Time step for dE/dt 
    def expStep(S, E, I, alpha, beta, lambd, my, omega, t, N, diff, h):
        #return E + ((((1 + omega*np.cos((2*np.pi*t*365)))*beta)*S*I)/N + (diff*S)/N - (my*E) - (alpha*E))*h
        return E + ((((1 + omega*np.cos(2*np.pi*t))*beta)*S*I)/N + (diff*S)/N - (my*E) - (alpha*E))*h

    #Time step for dI/dt 
    def infStep(E, I, alpha, beta, gamma, my, omega, t, N, diff, h):
        return I + (alpha*E - gamma*I - my*I)*h

    #Time step for dR/dt 
    def remStep(I, R, gamma, my, delta, h):
        return R + (gamma*I - my*R - delta*R)*h

    #Time step for Euler-method
    def timeStep(susMat, expMat, infMat, remMat, alpha, beta, gamma, lambd, my, delta, omega, t, N, m, h):
        tempSus = np.copy(susMat)
        tempExp = np.copy(expMat)
        tempInf = np.copy(infMat)
        tempRem = np.copy(remMat)

        if m != 0:
            diff = m * conv(tempInf)
        else:
            diff = 0

        tempSus = susStep(susMat, infMat, remMat, beta, lambd, my, delta, omega, t, N, diff, h)
        tempExp = expStep(susMat, expMat, infMat, alpha, beta, lambd, my, omega, t, N, diff, h)
        tempInf = infStep(expMat, infMat, alpha, beta, gamma, my, omega, t, N, diff, h)
        tempRem = remStep(infMat, remMat, gamma, my, delta, h)

        return tempSus, tempExp, tempInf, tempRem

    ## -------- Set up -------- ##

    susMat = np.zeros((xSize, ySize, iters))
    expMat = np.zeros((xSize, ySize, iters))
    infMat = np.zeros((xSize, ySize, iters))
    remMat = np.zeros((xSize, ySize, iters))

    susMat[:, :, 0] = np.ones((xSize, ySize)) * N
    infMat[:,:,0] = susMat[:,:,0]*inf
    susMat[:,:,0] -= infMat[:,:,0]

    if verbal == 1:
        print("Sus: \n", susMat[:, :, 0])
        print("Exp: \n", expMat[:, :, 0])
        print("Inf: \n", infMat[:, :, 0])
        print("Rem: \n", remMat[:, :, 0])
        print("alpha: \n", alpha)
        print("beta: \n", beta)
        print("my: \n", my)
        print("delta: \n", delta)
        print("omega: \n", omega)
        print("lambd: \n", lambd)
        print("m: \n", m)
        print("h: \n", h)
        print()

    ## -------- Run simulation -------- ##
    t = 0
    for i in range(iters-1):
        tempSus, tempExp, tempInf, tempRem = timeStep(susMat[:,:,i], expMat[:,:,i], infMat[:,:,i], remMat[:,:,i], alpha, beta, gamma, lambd, my, delta, omega, t, N, m, h)
        t += h

        if verbal == 2:
            print("\nSus: \n", tempSus)
            print("Exp: \n", tempExp)
            print("Inf: \n", tempInf)
            print("Rem: \n", tempRem)

        susMat[:,:,(i+1)] = tempSus
        expMat[:,:,(i+1)] = tempExp
        infMat[:,:,(i+1)] = tempInf
        remMat[:,:,(i+1)] = tempRem

        N = tempSus + tempExp + tempInf + tempRem

    if verbal == 1:
        print("Sus: \n", susMat[:, :, -1])
        print("Exp: \n", expMat[:, :, -1])
        print("Inf: \n", infMat[:, :, -1])
        print("Rem: \n", remMat[:, :, -1])

    return susMat, expMat, infMat, remMat

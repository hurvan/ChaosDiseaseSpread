import numpy as np
import cv2
import copy
import SIRSs_model as SIRSs_m
import SEIRS_model as SEIRSs_m

#To do: Resturcture so all parameters are given in constructor with defualt values for model-specific parameters set to zero.
#       Then the different methods for each model can exchanged for a single one that just runs the right model depending om the given parameters.
class run_epidemic_models:

    def __init__(self, iters):
        self.iters = iters

    def SIR(self, xSize, ySize, inf, beta, gamma, lambd, my, m, N):
        N = np.ones((xSize, ySize)) * N
        susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, 0, 0, m, N, inf)

        return susMat, infMat, remMat

    def SIR_load_population(self, xSize, ySize, inf, beta, gamma, lambd, my, m, filename):
        N = np.loadtxt(filename, skiprows=6)
        susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, 0, 0, m, N, inf)

        return susMat, infMat, remMat


    def SIRS(self, xSize, ySize, inf, beta, gamma, lambd, my, delta, m, N):
        self.N = np.ones((xSize, ySize)) * N
        susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, 0, m, N, inf)

        return susMat, infMat, remMat

    #SIRS model with a seasonality factor of spread(omega)
    def SIRSs(self, xSize, ySize, inf, beta, gamma, lambd, my, delta, omega, m, N):
        N = np.ones((xSize, ySize)) * N
        susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, omega, m, N, inf)

        return susMat, infMat, remMat

    #SIRS model with a seasonality factor of spread(omega)
    def SEIRSs(self, xSize, ySize, inf, alpha, beta, gamma, lambd, my, delta, omega, m, N):
        N = np.ones((xSize, ySize)) * N
        susMat, expMat, infMat, remMat = SEIRSs_m.SEIRSs_model(self.iters, xSize, ySize, alpha, beta, gamma, lambd, my, delta, omega, m, N, inf)

        return susMat, expMat, infMat, remMat

    #Test SIC for initial infected. Mobility factor set to zero.
    def SIC_test_inf(self, xSize, ySize, beta, gamma, lambd, my, delta, omega, N, inf1, inf2, alpha=0):
        inf = np.linspace(inf1, inf2, (xSize*ySize))
        inf = np.reshape(inf, (xSize, ySize))
        if alpha == 0:
            susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, omega, 0, N, inf)
            return susMat, infMat, remMat, inf
        else:
            susMat, expMat, infMat, remMat = SEIRSs_m.SEIRSs_model(self.iters, xSize, ySize, alpha, beta, gamma, lambd, my, delta, omega, 0, N, inf)
            return susMat, expMat, infMat, remMat, inf




    #Test parameter sensitivty for population. Mobility factor set to zero.
    def param_seni_test_N(self, xSize, ySize, inf, beta, gamma, lambd, my, delta, omega, N1, N2, alpha=0):
        N = np.linspace(N1, N2, (xSize*ySize))
        N = np.reshape(N, (xSize, ySize))
        inf = np.ones((xSize, ySize)) * 0.001
        if alpha == 0:
            susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, omega, 0, N, inf)
            return susMat, infMat, remMat, N
        else:
            susMat, expMat, infMat, remMat = SEIRSs_m.SEIRSs_model(self.iters, xSize, ySize, alpha, beta, gamma, lambd, my, delta, omega, 0, N, inf)
            return susMat, expMat, infMat, remMat, N

    #Test parameter sensitivty for population. Mobility factor set to zero.
    def param_seni_test(self, xSize, ySize, inf, beta, gamma, lambd, my, delta, omega, N, para1, para2, paraChoise, alpha=0):
        para = np.linspace(para1, para2, (xSize*ySize))
        para = np.reshape(para, (xSize, ySize))

        print("Sweep perameters: \n", para)
        if paraChoise == 1:
            beta = para
        elif paraChoise == 2:
            gamma = para
        elif paraChoise == 3:
            lambd = para
        elif paraChoise == 4:
            my = para
        elif paraChoise == 5:
            delta = para
        elif paraChoise == 6:
            omega = para
        elif paraChoise == 7 and alpha != 0:
            alpha = para
        else:
            print("Please choose parameter: 1 = beta, 2 = gamma, 3 = lambd, 4 = my, 5 = delta, 6 = omega")
            return
        if alpha.all() == 0:
            susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, omega, 0, N, inf)
            return susMat, infMat, remMat, para
        else:
            susMat, expMat, infMat, remMat = SEIRSs_m.SEIRSs_model(self.iters, xSize, ySize, alpha, beta, gamma, lambd, my, delta, omega, 0, N, inf)
            return susMat, expMat, infMat, remMat, para

    #Test parameter sensitivty for population. Mobility factor set to zero.
    def param_seni_test_m(self, xSize, ySize, inf, beta, gamma, lambd, my, delta, omega, N, para1, para2):
        m = np.linspace(para1, para2, (xSize*ySize))
        m = np.reshape(m, (xSize, ySize))

        print("Sweep perameters: \n", m)

        susRet = np.zeros((xSize, ySize, self.iters))
        infRet = np.zeros((xSize, ySize, self.iters))
        remRet = np.zeros((xSize, ySize, self.iters))

        for i in range(xSize):
            for j in range(ySize):
                susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, omega, m[i,j], N, inf)
                susRet[i,j,:] = np.sum(np.sum(susMat, axis=0), axis=0)
                infRet[i, j,:] = np.sum(np.sum(infMat, axis=0), axis=0)
                remRet[i, j,:] = np.sum(np.sum(remMat, axis=0), axis=0)

        return susRet, infRet, remRet, m

    #Test parameter sensitivty for population. Mobility factor set to zero.
    def param_seni_test_h(self, xSize, ySize, inf, beta, gamma, lambd, my, delta, omega, N, para1, para2):
        h = np.linspace(para1, para2, (xSize*ySize))
        h = np.reshape(h, (xSize, ySize))

        print("Sweep perameters: \n", h)
        susMat, infMat, remMat = SIRSs_m.SIRSs_model_h(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, omega, 0, N, inf, h)

        return susMat, infMat, remMat, h

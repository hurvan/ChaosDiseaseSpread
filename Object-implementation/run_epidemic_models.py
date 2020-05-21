import numpy as np
import cv2
import copy
import SIRSs_model as SIRSs_m

class run_epidemic_models:

    def __init__(self, iters):
        self.iters = iters

    def SIR(self, xSize, ySize, beta, gamma, lambd, my, m, N):
        self.N = np.ones((xSize, ySize)) * N
        susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, 0, 0, m, N)

        return susMat, infMat, remMat

    def SIR_load_population(self, xSize, ySize, beta, gamma, lambd, my, m, filename):
        N = np.loadtxt(filename, skiprows=6)
        susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, 0, 0, m, N)

        return susMat, infMat, remMat


    def SIRS(self, xSize, ySize, beta, gamma, lambd, my, delta, m, N):
        self.N = np.ones((xSize, ySize)) * N
        susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, 0, m, N, self.inf)

        return susMat, infMat, remMat

    #SIRS model with a seasonality factor of spread(omega)
    def SIRSs(self, xSize, ySize, beta, gamma, lambd, my, delta, omega, m, N):
        self.N = np.ones((xSize, ySize)) * N
        susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, omega, m, N, self.inf)

        return susMat, infMat, remMat

    #Test SIC for population. Mobility factor set to zero.
    def SIC_test_N(self, xSize, ySize, beta, gamma, lambd, my, delta, omega, N1, N2):
        self.N = np.linspace(N1, N2, (xSize*ySize))
        self.N = np.reshape(self.N, (xSize, ySize))
        self.inf = np.ones((xSize, ySize)) * 0.001
        susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, omega, 0, self.N, self.inf)

        return susMat, infMat, remMat

    #Test SIC for initial infected. Mobility factor set to zero.
    def SIC_test_inf(self, xSize, ySize, beta, gamma, lambd, my, delta, omega, N, inf1, inf2):
        self.inf = np.linspace(inf1, inf2, (xSize*ySize))
        self.inf = np.reshape(self.inf, (xSize, ySize))
        susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, omega, 0, N, self.inf)

        return susMat, infMat, remMat

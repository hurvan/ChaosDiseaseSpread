import numpy as np
import cv2
import copy
import SIR_model as SIR_m
import SIRS_model as SIRS_m
import SIRSs_model as SIRSs_m

class run_epidemic_models:

    def __init__(self, iters):
        self.iters = iters

    def SIR(self, xSize, ySize, beta, gamma, lambd, my, m, N):
        self.N = np.ones((xSize, ySize)) * N
        susMat, infMat, remMat = SIR_m.SIR_model(self.iters, xSize, ySize, beta, gamma, lambd, my, m, N)

        return susMat, infMat, remMat

    def SIR_load_population(self, xSize, ySize, beta, gamma, lambd, my, m, filename):
        N = np.loadtxt(filename, skiprows=6)
        susMat, infMat, remMa = SIR_m.SIR_model(self.iters, xSize, ySize, beta, gamma, lambd, my, m, N)

        return susMat, infMat, remMat


    def SIRS(self, xSize, ySize, beta, gamma, lambd, my, delta, m, N):
        self.N = np.ones((xSize, ySize)) * N
        susMat, infMat, remMat = SIRS_m.SIRS_model(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, m, N)

        return susMat, infMat, remMat

    #SIRS model with a seasonality factor of spread(omega)
    def SIRSs(self, xSize, ySize, beta, gamma, lambd, my, delta, omega, m, N):
        self.N = np.ones((xSize, ySize)) * N
        susMat, infMat, remMat = SIRSs_m.SIRSs_model(self.iters, xSize, ySize, beta, gamma, lambd, my, delta, omega, m, N)

        return susMat, infMat, remMat

import plotter as p
import numpy as np
import run_epidemic_models as rem

#Simulation parameters
xSize = 10          #Size of grid in x direction
ySize = 10          #Size of grid in y direction
iters = 50000       #Number of itterations
alpha = 0.2           #Rate of infection from incubation phase 
beta = 0.25         #Mean contact * propability of infection
gamma = 0.1         #Rate of removal from infection(recovery + death)
lambd = 0           #Birth rate. Sweden: 0.000032
my = 0              #Death rate. Sweden: 0.000025
delta = 0.005       #Rate of imunnity loss
omega = 2           #Seasonality factor affecting spread
m = 0.005          #Mobility
N = 1000            #Population
inf1 = np.zeros((xSize, ySize))     #Initaly infected, proportion
inf1[0, 0] = 0.05

#SIC and parameter sensitivity sweep parameters
N1 = 1000
N2 = 5000
infP1 = 0
infP2 = 0.9
para1 = 0.000001
para2 = 4
inf2 = np.ones((xSize, ySize))*0.005     #Initaly infected
paraChoise = 7      #Parameter to sweep: 1 = beta, 2 = gamma, 3 = lambd, 4 = my, 5 = delta, 6 = omega

model = rem.run_epidemic_models(iters)

#susMat, infMat, remMat = model.SIR(xSize, ySize, inf1, beta, gamma, lambd, my, m, N)
#susMat, infMat, remMat = model.SIRS(xSize, ySize, inf1, beta, gamma, lambd, my, delta, m, N)
#susMat, infMat, remMat = model.SIRSs(xSize, ySize, inf1, beta, gamma, lambd, my, delta, omega, m, N)
#susMat, expMat, infMat, remMat = model.SEIRSs(xSize, ySize, inf1, alpha, beta, gamma, lambd, my, delta, omega, m, N)

#susMat, infMat, remMat, parRange = model.param_seni_test_N(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N1, N2)
#susMat, infMat, remMat, parRange = model.SIC_test_inf(xSize, ySize, beta, gamma, lambd, my, delta, omega, N, infP1, infP2)
#susMat, infMat, remMat, parRange = model.param_seni_test(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N, para1, para2, paraChoise)
#susMat, infMat, remMat, parRange = model.param_seni_test_h(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N, para1, para2)

#susMat, expMat, infMat, remMat, parRange = model.SIC_test_inf(xSize, ySize, beta, gamma, lambd, my, delta, omega, N, infP1, infP2, alpha)
#susMat, expMat, infMat, remMat, parRange = model.param_seni_test_N(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N1, N2, alpha)
susMat, expMat, infMat, remMat, parRange = model.param_seni_test(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N, para1, para2, paraChoise, alpha)

#Warning, a lot of compuations
#susMat, infMat, remMat, parRange = model.param_seni_test_m(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N, para1, para2)

plotter = p.plotter(susMat, infMat, remMat)
#plotter.animate_all()
plotter.animate_all_SERISs(expMat)
#plotter.plot_all()
#plotter.animate_inf_sus()
#plotter.phase_diagrams(0, 0)
#plotter.biforcation_diagram(100, parRange)
#plotter.plot_all_h(parRange)

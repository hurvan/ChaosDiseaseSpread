import plotter as p
import numpy as np
import run_epidemic_models as rem

# This is my first time writing in python so some solutions may be strange. A gui-would be a nice continouation. The different functions in
# run_epidimic_models could be compressed and run denpendent on which parameters are given with use of default parameters.
# The use of multipule main-programs is always a good way to get bugs, but this is now set to parameters which give chaos 
# where as the other uses some common ones.
# This program very easily fills ones RAM so a way to deal with that would also be nice

save = True
load = False

#Simulation parameters
h = 0.001
xSize = 15                              # Size of grid in x direction
ySize = 15                              # Size of grid in y direction
iters = 50000                           # Number of itterations
alpha = np.ones((1, 1)) * 35.48         # Rate of infection from incubation phase 
beta = 1800                             # Mean contact * propability of infection
gamma = 100                             # Rate of removal from infection(recovery + death)
lambd = 0.02                            # Birth rate. Sweden: 0.000032
my = 0.02                               # Death rate. Sweden: 0.000025
delta = 0                               # Rate of imunnity loss
omega = 0.28                            # Seasonality factor affecting spread
m = 0                                   # Mobility
N = 1                                   # Population
inf1 = np.zeros((xSize, ySize))         # Initaly infected, proportion
inf1[0, 0] = 1/100

#SIC and parameter sensitivity sweep parameters
N1 = 1000
N2 = 5000
infP1 = 0.01
infP2 = 0.011
para1 = 0.00001
para2 = 200
inf2 = np.ones((xSize, ySize))*1/100    # Initaly infected
paraChoise = 7                          # Parameter to sweep: 1 = beta, 2 = gamma, 3 = lambd, 4 = my, 5 = delta, 6 = omega, 7 = alpha

model = rem.run_epidemic_models(iters)

#susMat, infMat, remMat = model.SIR(xSize, ySize, inf1, beta, gamma, lambd, my, m, N)
#susMat, infMat, remMat = model.SIRS(xSize, ySize, inf1, beta, gamma, lambd, my, delta, m, N)
#susMat, infMat, remMat = model.SIRSs(xSize, ySize, inf1, beta, gamma, lambd, my, delta, omega, m, N)
#susMat, expMat, infMat, remMat = model.SEIRSs(xSize, ySize, inf1, alpha, beta, gamma, lambd, my, delta, omega, m, N, h)

#susMat, infMat, remMat, parRange = model.param_seni_test_N(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N1, N2)
#susMat, infMat, remMat, parRange = model.SIC_test_inf(xSize, ySize, beta, gamma, lambd, my, delta, omega, N, infP1, infP2)
#susMat, infMat, remMat, parRange = model.param_seni_test(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N, para1, para2, paraChoise)
#susMat, infMat, remMat, parRange = model.param_seni_test_h(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N, para1, para2)

#susMat, expMat, infMat, remMat, parRange = model.SIC_test_inf(xSize, ySize, beta, gamma, lambd, my, delta, omega, N, infP1, infP2, alpha, h)
#susMat, expMat, infMat, remMat, parRange = model.param_seni_test_N(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N1, N2, alpha, h)
susMat, expMat, infMat, remMat, parRange = model.param_seni_test(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N, para1, para2, paraChoise, alpha, h)
#susMat, expMat, infMat, remMat, parRange = model.param_seni_test_h(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N, para1, para2, alpha)

## Warning, a lot of compuations
#susMat, infMat, remMat, parRange = model.param_seni_test_m(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N, para1, para2, h)
#susMat, expMat, infMat, remMat, parRange = model.param_seni_test_m(xSize, ySize, inf2, beta, gamma, lambd, my, delta, omega, N, para1, para2, alpha, h)

if save == True:
    np.save("susMat", susMat)
    np.save("expMat", expMat)
    np.save("infMat", infMat)
    np.save("remMat", remMat)
if load == True:
    susMat = np.load("susMat.npy")
    expMat = np.load("expMat.npy")
    infMat = np.load("infMat.npy")
    remMat = np.load("remMat.npy")
    parRange = np.linspace(para1, para2, xSize*ySize)
    parRange = np.reshape(parRange, (xSize, ySize))
    
#susMat = susMat[:, :, 1000:-1]
#infMat = infMat[:, :, 1000:-1]
#remMat = remMat[:, :, 1000:-1]
#expMat = expMat[:, :, 1000:-1]

plotter = p.plotter(susMat, infMat, remMat)
#plotter.animate_all()
#plotter.animate_all_SERISs(expMat)
#plotter.plot_all()
#plotter.plot_all_SEIR(expMat)
#plotter.plot_all_N(parRange, expMat)
#plotter.animate_inf_sus()
#plotter.phase_diagrams(0, 0)
#plotter.phase_diagramsAll(expMat)
plotter.biforcation_diagram(10, parRange, expMat)
#plotter.plot_all_h(parRange, expMat)

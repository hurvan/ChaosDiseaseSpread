import plotter as p
import run_epidemic_models as rem

xSize = 10          #Size of grid in x direction
ySize = 10          #Size of grid in y direction
iters = 80000       #Number of itterations
beta = 0.25         #Mean contact * propability of infection
gamma = 0.1         #Rate of removal from infection(recovery + death)
lambd = 0.000032    #Birth rate 
my = 0.000025       #Death rate
delta = 0.001       #Rate of imunnity loss
omega = 1         #Seasonality factor affecting spread
m = 0.0005          #Mobility
N = 1000            #Population

#SIC perameters
N1 = 1000
N2 = 1000.01
inf1 = 0.001
inf2 = 0.0012

model = rem.run_epidemic_models(iters)

#susMat, infMat, remMat = model.SIR(xSize, ySize, beta, gamma, lambd, my, m, N)
#susMat, infMat, remMat = model.SIRS(xSize, ySize, beta, gamma, lambd, my, delta, m, N)
#susMat, infMat, remMat = model.SIRSs(xSize, ySize, beta, gamma, lambd, my, delta, omega, m, N)
#susMat, infMat, remMat = model.SIC_test_N(xSize, ySize, beta, gamma, lambd, my, delta, omega, N1, N2)
susMat, infMat, remMat = model.SIC_test_inf(xSize, ySize, beta, gamma, lambd, my, delta, omega, N, inf1, inf2)

plotter = p.plotter(susMat, infMat, remMat)
plotter.animate_all()
#plotter.plot_all()
#plotter.animate_inf_sus()

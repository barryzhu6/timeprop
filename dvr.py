"""
Created on Mon Mar  6 21:06:32 2023

@author: zhujk
"""

import numpy as np
import matplotlib.pyplot as plt

m = 1000
pi = np.pi
x_min = -7
x_max = 7
N = 1000
delta = (x_max-x_min)/(N+1)

#change potential function here
def Vfunc(x):
    # if x < -1.5*pi:
    #     return -0.002*np.sin(-1.5*pi)*(x+1.5*pi)
    # if -1.5*pi <= x <= 1.5*pi:
    #     return np.cos(x)*0.002
    # if x > 1.5*pi:
    #     return -0.002*np.sin(1.5*pi)*(x-1.5*pi)
    return 0.5 * m * 0.0027338**2 * x**2

T = np.zeros((N,N))
V = np.zeros((N,N))
cor = np.zeros(N)
Vgraph = np.zeros(N)
for i in range(N):
    T[i,i] = (i+1)**2 * pi**2 /2 /m /(x_max-x_min)**2
    cor[i] = x_min + (i+1)*delta
    V[i,i] = Vfunc(cor[i])
    Vgraph[i] = Vfunc(cor[i])
U = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        U[i,j] = np.sqrt(2/(N+1)) * np.sin((i+1)*(j+1)*pi/(N+1))
H = V + U.dot(T).dot(U)

e, v = np.linalg.eigh(H)
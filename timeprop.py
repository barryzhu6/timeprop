# -*- coding: utf-8 -*-
"""
VS code

split-operator time propagation
"""

import numpy as np
import cmath as cm
from scipy.special import eval_hermite as herm
import matplotlib.pyplot as plt
import matplotlib.animation as animi
import math

m = 1000
pi = np.pi
x_min = -7
x_max = 7
N = 1000
delta = (x_max - x_min)/(N+1)
t = 1

def time_exp(x:complex) -> complex:
    return cm.exp(-1j * x * t / 1)

def Tf(x:int) -> complex:
    return x**2 * pi**2 / 2 / m / (x_max-x_min)**2

def Vf(x:float) -> complex:
    return 0.5 * m * 0.0027338**2 * x**2

def initwfn(x,n):
    return 1/(np.sqrt(2**n * math.factorial(n))) * (m*0.0027338/pi)**0.25 \
        * np.exp(-m*0.0027338* (x) **2/2) * herm(n,np.sqrt(m*0.0027338)* (x) )

wfn = np.zeros(N)

T = np.zeros((N,N),dtype=complex)
for i in range(N):
    T[i,i] = time_exp(Tf(i+1))
U = np.zeros((N,N))
for n in range(N):
    for m in range(N):
        U[n,m] = np.sqrt(2/(N+1)) * np.sin((m+1)*(n+1)*pi/(N+1))
V = np.zeros((N,N),dtype=complex)
Vshow = np.zeros(N)
cor = np.zeros(N)
for i in range(N):
    cor[i] = x_min + (i+1) * delta
    V[i,i] = time_exp(Vf(cor[i]) / 2)
    Vshow[i] = Vf(cor[i])

# change wfn here
    wfn[i] = (initwfn(cor[i],0)/np.sqrt(2) + initwfn(cor[i],1)/np.sqrt(2))

M = np.linalg.multi_dot([V,U,T,U,V])




fig, ax = plt.subplots()

# probabilities times 10 to make it visible
line, = ax.plot(cor, abs(wfn)**2 * delta * 10)
#line, = ax.plot(cor, wfn)

def animate(i):
    global wfn
    for j in range(10):
        wfn = M.dot(wfn)
    line.set_ydata(abs(wfn)**2 * delta * 10)
#    line.set_ydata(wfn)
    return line, 

ani = animi.FuncAnimation(
    fig, animate, interval = 20, blit = True, save_count= 50)


ax.plot(cor, Vshow)
plt.show() 
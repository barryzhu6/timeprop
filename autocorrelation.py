# -*- coding: utf-8 -*-
"""
VS code

split-operator time propagation
with autocorrelation
"""

import numpy as np
import cmath as cm
from scipy.special import eval_hermite as herm
import matplotlib.pyplot as plt
import math
import torch
from scipy.signal import argrelextrema
import dvr

m = 1000
pi = np.pi
x_min = -7
x_max = 7
N = 1000
delta = (x_max - x_min)/(N+1)
t_step = 10
step:int = 1000001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def time_exp(x:complex) -> complex:
    return cm.exp(-1j * x * t_step / 1)

def Tf(x:int) -> complex:
    return x**2 * pi**2 / 2 / m / (x_max-x_min)**2

def Vf(x:float) -> complex:
    return 0.5 * m * 0.0027338**2 * x**2

def initwfn(x,n:int):
    return 1/(np.sqrt(2**n * math.factorial(n))) * (m*0.0027338/pi)**0.25 \
        * np.exp(-m*0.0027338* (x) **2/2) * herm(n,np.sqrt(m*0.0027338)* (x) )

def localmaxes(a):
    out = []
    for i in range(len(a)):
        if(i == 0 or i == (len(a)-1)):
            continue
        if(a[i-1]<a[i] and a[i]>a[i+1]):
            out.append(i)
    return out


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
# modify wfn here
    # wfn[i] = initwfn(cor[i],0)/np.sqrt(3) + initwfn(cor[i],1)/np.sqrt(3) + initwfn(cor[i],2)/np.sqrt(3)
wfn = dvr.v[:,0]/np.sqrt(2) + dvr.v[:,1]/np.sqrt(2)
M = np.linalg.multi_dot([V,U,T,U,V])
wfn_T = torch.from_numpy(wfn).type(torch.complex128).to(device)
M_T = torch.from_numpy(M).type(torch.complex128).to(device)

A = np.zeros(step, dtype=complex)
ω = np.zeros(step)
t = np.zeros(step)

wfnt = torch.clone(wfn_T)

for i in range(step):
    wfn_T = M_T @ wfn_T
    t[i] = (i+1)*t_step
    A[i] = torch.linalg.vecdot(wfn_T,wfnt).item() * np.exp(-0.5*((i+1)*t_step/2400000)**2)

fftA = np.fft.fft(A)
ω = np.fft.fftfreq(step, t_step) * (2*pi)

ω = np.concatenate((ω[int(step/2)+1:],ω[:int(step/2)+1]))
fftA = np.concatenate((fftA[int(step/2)+1:],fftA[:int(step/2)+1]))

fig, axs = plt.subplots(1,2)
axs[0].plot(t,abs(A))
axs[1].plot(ω,abs(fftA))
maxs = argrelextrema(fftA, np.greater)[0]
for i in range(len(maxs)):
    print(ω[maxs[i]])
plt.show()
print("ref:")
print(dvr.e[0])
print(dvr.e[1])
print(len(maxs))
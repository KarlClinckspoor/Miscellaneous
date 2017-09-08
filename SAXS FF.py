# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 13:17:45 2017

@author: Karl
"""

import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy import integrate

#%%
def sphere_FF (q, R):

    FF = ((3*(sin(q*R)-q*R*cos(q*R)) / (q*R)**3)**2)
    return FF

#%%
q = np.logspace(-2,1,num=250)
r = [1,10,100]

for item in r:
    FF = sphere_FF(q,item)
    plt.plot(q, FF)


plt.xscale('log')
plt.yscale('log')
plt.show()


#%%
def JobBessel():
    orders = [0,1,2,3,4,5]
    x = np.linspace(0,10,500)
    for order in orders:
        plt.plot(x, jv(order, x))

    plt.show()

JobBessel()
#%% Grandes chances de que essa equação está errada, tem que dar uma olhada melhor em outros artigos
def Cylinder_FF_to_integrate (q, R, L, alpha):
    
    
    first_term = (2*jv(1, q*R*sin(alpha)) / q*R*sin(alpha))
    second_term = (sin(q*L*cos(alpha/2))/q*L*cos(alpha/2))
    FF = (first_term*second_term)**2*sin(alpha)
    return FF


#==============================================================================
#      numerator = (2*jv(1, (q*R*sin(alpha))) * sin(q*L*cos(alpha/2)))
#      denominator = (q*R*sin(alpha) * q*L*cos(alpha/2))
#      FF = (numerator/denominator)**2*sin(alpha)
#      return FF
#==============================================================================
#%%
def Cylinder_FF_to_integrate2 (q, R, L, al):
    j = sin(q*L/2*cos(al)) / (q*L/2*cos(al))
    f = j*jv(1, q*R*sin(al))/(q*R*sin(al))
    to_int = f**2*sin(al)
    return to_int*500   

#%%

def Cylinder_FF_to_integrate3 (q, R, L, al):
    qrsin = q*R*sin(al)
    qlcos = q*L*cos(al)/2
    
    value = ( 2*jv(1, qrsin)/qrsin * sin(qlcos)/qlcos )**2 * sin(al)
    return value

#%%
length = 500
qs = np.logspace(-2,0,num=length)
R = 30
L = 120
alphas = np.linspace(1e-2, np.pi/2, num=length)
Int = np.zeros(length)
y_simps = np.zeros(length)

for i, q in enumerate(qs):
    #Int[i] = sum(Cylinder_FF_to_integrate3(q, R, L, alpha) for alpha in alphas)
    #Int[i] = integrate.quad(Cylinder_FF_to_integrate3, 1e02, np.pi/2, args = (q, R, L))
    for j, alpha in enumerate(alphas):
        y_simps[j] = Cylinder_FF_to_integrate3(q, R, L, alpha)
    Int[i] = integrate.simps(y_simps, x=alphas)
    

plt.xscale('log')
plt.yscale('log')
plt.plot(qs, Int)
plt.show()

#%%

def Guinier (I0, q, Rg):
    return I0*np.exp(-(q**2*Rg**2)/3)

q = np.logspace(1E-3,1,num=500)

Rg = 3
I0 = 3

G = Guinier(I0, q, Rg)

q_mod = q**2
G_mod = np.log(G)

plt.plot(q, G)
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.plot(q_mod, G_mod)
plt.show()

#%%
def r_f (R, Ep, Th):
    return R*(sin(Th)**2+Ep**2*cos(Th)**2)**(1/2)

def Ellipsoid_to_int (q, R, Ep, Th):
    r = r_f(R, Ep, Th)
    Num = 3 * ( sin(q*r) - q*r*cos(q*r) )
    Den = (q*r)**3
    to_int = (Num/Den)**2*sin(Th)
    return to_int




'''
R = [1, 10, 100]
EpR = [0.2, 0.5, 1, 3]
'''
#%%
length = 250
qs = np.logspace(0,1, num=length)
R = 1
#EpR = 100
Ep = 1
Int = np.zeros(length)
Thetas_temp = np.logspace(1E-2, np.pi/2, num=length)

for i, q in enumerate(qs):
   Int[i] = sum(Ellipsoid_to_int(q, R, Ep, Theta) for Theta in Thetas_temp)
   #Int[i], err = integrate.quad(Ellipsoid_to_int, 1E-2, np.pi/2, args = (q, R, Ep))

#plt.figure(figsize=(20,10))
plt.plot(qs, Int)
plt.plot(qs, sphere_FF(qs, R))
plt.xscale('log')
plt.yscale('log')


plt.show()
#
length = 150
q = np.logspace(-2, 0, num=length)
R = 1
EP = 1
L = 150

Int_sphere = np.zeros(length)
Int_sphere = np.zeros(length)
Int_sphere = np.zeros(length)
    
    
    
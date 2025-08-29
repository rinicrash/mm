# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 23:40:28 2025

@author: shwet
"""

#example to integrate to obtain exact value
import sympy as sp

x = sp.symbols('x')
f = (1 + 2*x) * sp.sqrt(x)   # example f(x)

# definite integral from 0 to 2
res = sp.integrate(f, (x, 0, 2))
print("Exact integral:", res.evalf())




import numpy as np
#trapezoidal
def f(x):
    return np.sin(x)

def trapezoidal(a,b,n):
    h=(b-a)/(n-1)
    x=np.linspace(a,b,n)
    y=f(x)
    area= (h/2)*(y[0] + 2*np.sum(y[1:n-1]) + y[n-1])
    print("Area:",area)
    
def simpson(a,b,n):
    if n % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of points (even subintervals).")
    h = (b-a)/(n-1)
    x = np.linspace(a,b,n)
    y = f(x)
    area = (h/3)*(y[0] + y[-1] + 4*np.sum(y[1:n-1:2]) + 2*np.sum(y[2:n-2:2]))
    print("Simpson Area:", area)

    
trapezoidal(0,5,10)
simpson(0,5,9)

    
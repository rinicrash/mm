# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 23:18:56 2025

@author: shwet
"""

import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

x=sp.symbols('x')
y=sp.Function('y')
ode=sp.Eq(sp.diff(y(x),x,2) + 0.6*sp.diff(y(x),x) + 8*y(x),0)
sol=sp.dsolve(ode)

x=[0]
y=[4]
yp=[0]

# Analytical solution
def analytical_solution(x):
    return np.exp(-0.3 * x) * (C1 * np.cos(beta * x) + C2 * np.sin(beta * x))


def derivatives(x0,y0,z0):
    dy1_dx=z0
    dy2_dx=(-0.6*z0 )+ (-8*y0)
    return dy1_dx,dy2_dx

from scipy.stats import t

def paired_ttest(x, y):
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Differences
    d = x - y
    n = len(d)

    # Mean and std of differences
    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)  # sample standard deviation

    # t-statistic
    t_stat = d_mean / (d_std / np.sqrt(n))

    # Two-tailed p-value
    p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n-1))

    return t_stat, p_value



h=0.5
x_start=0
x_end=5

beta = math.sqrt(7.91)
C1 = 4
C2 = 1.2 / beta


while x[-1] < x_end:
    x_current=x[-1]
    y_current=y[-1]
    yP_current=yp[-1]
    
    k1_y, k1_yP = derivatives(x_current, y_current, yP_current)

    # k2
    x_k2 = x_current + h/2
    y_k2 = y_current + h/2 * k1_y
    yP_k2 = yP_current + h/2 * k1_yP
    k2_y, k2_yP = derivatives(x_k2, y_k2, yP_k2)

    # k3
    x_k3 = x_current + h/2
    y_k3 = y_current + h/2 * k2_y
    yP_k3 = yP_current + h/2 * k2_yP
    k3_y, k3_yP = derivatives(x_k3, y_k3, yP_k3)

    # k4
    x_k4 = x_current + h
    y_k4 = y_current + h * k3_y
    yP_k4 = yP_current + h * k3_yP
    k4_y, k4_yP = derivatives(x_k4, y_k4, yP_k4)

    # yn+1 = yn + h/6(k1+2k2+2k3+k4)
    y_next = y_current + h/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    yP_next = yP_current + h/6 * (k1_yP + 2*k2_yP + 2*k3_yP + k4_yP)
    x_next = x_current + h
    
    x.append(x_next)
    y.append(y_next)
    yp.append(yP_next)
    
orig=analytical_solution(np.array(x))

plt.plot(x,orig,"o--", label="Analytical")    
plt.plot(x, y, label='RK4 Method', marker='^', linestyle='-', color='green')
plt.legend()
plt.show()

print(paired_ttest(orig,y))
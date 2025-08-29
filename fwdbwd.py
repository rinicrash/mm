# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 18:29:04 2025

@author: shwet
"""

import numpy as np
import sympy as sp
import math
import pandas
import matplotlib.pyplot as plt
import pandas as pd

def diff_table(x,y,n):
    dd_table=np.zeros((n,n))
    dd_table[:,0]=y
    for j in range(1,n):
        for i in range(n-j):
            dd_table[i][j]=dd_table[i+1][j-1] - dd_table[i][j-1]
    return dd_table

def forward_interp(x,y,n,xi):
    table=diff_table(x,y,n)
    h=x[1]-x[0]
    p=(xi-x[0])/h
    f = y[0]
    term=1
    for j in range(1,n):
        term*=(p-j+1)
        f+= (term * table[0][j])/math.factorial(j)
    return f


def backward_interp(x,y,n,xi):
    table=diff_table(x,y,n)
    h=x[1]-x[0]
    p=(xi-x[-1])/h
    f = y[-1]
    term =1
    for j in range(1,n):
        term*=(p+j - 1)
        #print(table[n-j-1][j])
        f += (term * table[n-j-1][j])/math.factorial(j)
    return f

x_vals = np.linspace(0, 3*np.pi, 7)
y_vals = np.sin(x_vals) * np.exp(-x_vals/2)
x1 = 2.5
x2 = 8.5



df=pd.DataFrame(diff_table(x_vals,y_vals,len(x_vals)))
print(df)
val=forward_interp(x_vals,y_vals,len(x_vals),x1)
print(val)

val2 = backward_interp(x_vals,y_vals,len(x_vals),x2)
print(val2)

plt.plot(x_vals,y_vals)
plt.scatter([2.5,8.5],[val,val2])

y1_pred= np.array([forward_interp(x_vals,y_vals,len(x_vals),i) for i in x_vals ])
y2_pred = np.array([backward_interp(x_vals,y_vals,len(x_vals),i) for i in x_vals ])

#ttest
from scipy.stats import t
n=len(y_vals)
d= y_vals - y1_pred
mean=np.mean(d)
std=np.std(d,ddof=1)
ttest_val = mean /(std/np.sqrt(n))
p_val= 2* t.sf(np.abs(ttest_val),df=n-1)

alpha=0.05
#h0: no significant difference
if p_val<alpha:
    print("Reject H0: There is significant difference")
else:
    print("Accept H0: No significant difference")
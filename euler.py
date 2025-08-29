# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 20:06:51 2025

@author: shwet
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def exact(x):
    return ((x+(x**2)+2)/2)**2

def euler(x,y):
    return (1+2*x)*math.sqrt(y)

h=0.25
x=np.arange(0,1.25,0.25)
y_euler=[1]
y_exact=[]

#euler method (yi+1 = yi + f(xi,yi)*h)
for i in range(len(x)-1):
    yi=y_euler[-1]
    f=euler(x[i],yi)
    val=yi+(f*h)
    y_euler.append(val)
    
#exact
for i in range(len(x)):
    y_exact.append(exact(x[i]))
    

diff=np.array(y_euler)-np.array(y_exact)
rel=(diff/np.array(y_exact))*100

#table
df=pd.DataFrame(
    {
     "x":x,
     "y_exact":y_exact,
     "y_euler":y_euler,
     "Difference":diff,
     "Rel Error":rel
     })
    

plt.plot(x,y_euler,'o-',label="Euler",color="blue")
plt.plot(x,y_exact,'o-',label="Exact",color="red")
plt.legend()
plt.show()

print(df)

#test
from scipy import stats
#(mean1 - mean2)/[root(var1/n1) + (var2/n2)]
mean1=np.mean(y_euler)
mean2=np.mean(y_exact)
var1=np.var(y_euler,ddof=1)
var2=np.var(y_exact,ddof=1)
n1=len(y_euler)
n2=len(y_exact)
ttest = (mean1 - mean2) / math.sqrt((var1/len(y_euler)) + (var2/len(y_exact)))

# Step 3: Welch–Satterthwaite degrees of freedom
df = ((var1/n1 + var2/n2)**2) / (((var1/n1)**2)/(n1-1) + ((var2/n2)**2)/(n2-1))

# Step 4: Two-tailed p-value
p_value = 2 * (1 - stats.t.cdf(abs(ttest), df))

print("t-value:", ttest)
print("degrees of freedom:", df)
print("p-value:", p_value)






#suppose find polynomial from the data
import sympy as sp
from sympy import sympify,diff,lambdify,symbols

coeffs=np.polyfit(x,y_exact,10)
x_var=symbols('x')
poly=sum(coeff*x_var**i for i,coeff in enumerate(coeffs[::-1]))


derivative=sp.diff(poly,x_var)

print("poly:",poly)
print("derivative:",derivative)

#convert to function
deriv_function=sp.lambdify(x_var,derivative,'numpy')

y_euler1=[1]
#euler method (yi+1 = yi + f(xi,yi)*h)
for i in range(len(x)-1):
    yi=y_euler1[-1]
    f=deriv_function(x[i])
    val=yi+(f*h)
    y_euler1.append(val)
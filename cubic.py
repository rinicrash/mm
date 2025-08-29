# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 17:07:38 2025

@author: shwet
"""

import random
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

n = 4
x = [3,4.5,7,9]
f = [2.5,1,2.5,0.5]

def spline_equation_row(xi_minus1, xi, xi_plus1, f_xi_minus1, f_xi, f_xi_plus1):
    a = xi - xi_minus1
    b = 2 * (xi_plus1 - xi_minus1)
    c = xi_plus1 - xi
    rhs = (6 / (xi_plus1 - xi)) * (f_xi_plus1 - f_xi) - (6 / (xi - xi_minus1)) * (f_xi - f_xi_minus1)
    return a, b, c, rhs

variables = sp.symbols(f'f\'\'1:{n-1}')
equations = []

for i in range(1, n - 1):
    a, b, c, rhs = spline_equation_row(x[i - 1], x[i], x[i + 1], f[i - 1], f[i], f[i + 1])
    row = 0
    if i > 1:
        row += a * variables[i - 2]
    row += b * variables[i - 1]
    if i < n - 2:
        row += c * variables[i]
    equations.append(sp.Eq(row, rhs))

sol = sp.solve(equations, variables)
fpp = [0] + [sol[v] for v in variables] + [0]

def cubic_spline_interpolation(x_val, xi, xi1, fxi, fxi1, fppi, fppi1):
    h = xi1 - xi
    term1 = fppi * ((xi1 - x_val) ** 3) / (6 * h)
    term2 = fppi1 * ((x_val - xi) ** 3) / (6 * h)
    term3 = (fxi / h - fppi * h / 6) * (xi1 - x_val)
    term4 = (fxi1 / h - fppi1 * h / 6) * (x_val - xi)
    #print(f"{term1}+{term2}+{term3}+{term4}")
    return term1 + term2 + term3 + term4

X_vals = []

Y_vals = []

for i in range(n - 1):
    xs = np.linspace(x[i],x[i+1],100)
    ys = [cubic_spline_interpolation(xv, x[i], x[i + 1], f[i], f[i + 1], fpp[i], fpp[i + 1]) for xv in xs]
    X_vals.extend(xs)
    Y_vals.extend(ys)

plt.figure(figsize=(12, 6))
plt.plot(X_vals, Y_vals, label="Cubic Spline", color="blue")
plt.plot(x, f, 'ro', label="Data Points")
plt.title("Cubic Spline Interpolation")
plt.legend()
plt.grid(True)
plt.show()
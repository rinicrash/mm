# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 19:13:01 2025

@author: shwet
"""

import numpy as np
from scipy.stats import t

# Example data
before = np.array([100, 102, 98, 97, 105, 110, 108])
after  = np.array([102, 101, 100, 99, 107, 111, 109])

# Step 1: Differences
d = before - after

# Step 2: Mean difference
d_mean = np.mean(d)

# Step 3: Standard deviation of differences
d_std = np.std(d, ddof=1)  # sample std dev

# Step 4: Number of samples
n = len(d)

# Step 5: t-statistic
t_stat = d_mean / (d_std / np.sqrt(n))

# Step 6: Two-tailed p-value
p_val = 2 * t.sf(np.abs(t_stat), df=n-1)

# Step 7: Interpretation
alpha = 0.05
print("t-statistic:", t_stat)
print("p-value:", p_val)

if p_val < alpha:
    print(f"Reject H0 at α={alpha}: Significant difference between before and after.")
else:
    print(f"Fail to Reject H0 at α={alpha}: No significant difference between before and after.")

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:23:14 2018

@author: StepanAsus
"""

import numpy as np


#%%
# instructions
np.random.seed(1429)
n = 50
rep = 1000
boot = 499
m = 0
s = 1.3
mu = np.exp(m + 1 / 2 * s**2)
beta = np.sin(mu)

xbar = np.zeros((rep, 1)) # average of original sample
bhat = np.zeros((rep, 1)) # estiamte of beta
se = np.zeros((rep, 1)) # standard error bhat (asymptotic)
trat = np.zeros((rep, 1)) # t-ratio
lcl_asym = np.zeros((rep, 1)) # lower confidence limit (asym)
ucl_asym = np.zeros((rep, 1)) # upper confidence limit (asym)


#%%
# for loop from instructions + question 1
np.random.seed(1429)

for i in range(rep):
  if i % 100 == 0:
    print(i)
  x = np.exp(np.random.normal(m, s, (n, 1)))
  xbar[i] = np.mean(x)
  bhat[i] = np.sin(xbar[i])
  se[i] = np.sqrt(np.cos(xbar[i]) * np.var(x) / n * (np.cos(xbar[i])))
  trat[i] = (bhat[i] - beta) / se[i]
  lcl_asym[i] = bhat[i] - 1.96 * se[i]
  ucl_asym[i] = bhat[i] + 1.96 * se[i]

cov_freq_asym = np.mean(np.logical_and(beta > lcl_asym, beta < ucl_asym))

#%%
# question 2
np.random.seed(1429)

for n in range(50, 2500, 50):
  for i in range(0, rep):
    if i % 100 == 0:
      print(i)
    x = np.exp(np.random.normal(m, s, (n, 1)))
    xbar[i] = np.mean(x)
    bhat[i] = np.sin(xbar[i])
    se[i] = np.sqrt(np.cos(xbar[i]) * np.var(x) / n * (np.cos(xbar[i])))
    trat[i] = (bhat[i] - beta) / se[i]
    lcl_asym[i] = bhat[i] - 1.96 * se[i]
    ucl_asym[i] = bhat[i] + 1.96 * se[i]
  if 0.9365 < np.mean(np.logical_and(beta > lcl_asym, beta < ucl_asym)) < 0.9635:
    minimal = n
    break
    
#%%
# question 3
np.random.seed(1429)
n = 50
x = np.exp(np.random.normal(m, s, (n, 1)))

# bootstrap equivalents
xbar_b = np.zeros((rep, 1)) # average of original sample
bias_corr_b = np.zeros((rep, 1)) # estiamte of beta
bias_corr_jk = np.zeros((rep, 1)) # estiamte of beta
se_b = np.zeros((rep, 1)) # standard error bhat (asymptotic)
trat_b = np.zeros((rep, 1)) # t-ratio
trat_jk = np.zeros((rep, 1)) # t-ratio
lcl_asym_b = np.zeros((rep, 1)) # lower confidence limit (asym)
lcl_asym_jk = np.zeros((rep, 1)) # lower confidence limit (asym)
ucl_asym_b = np.zeros((rep, 1)) # upper confidence limit (asym)
ucl_asym_jk = np.zeros((rep, 1)) # upper confidence limit (asym)


for i in range(rep):
  x = np.exp(np.random.normal(m, s, (n, 1)))

  # bootstrap
  xbar_aux = np.zeros((boot, 1))
  aux_bhat = np.zeros((boot, 1))
  for k in range(boot):
    index = np.random.randint(0, n - 1, n)
    xbar_aux[k] = np.mean(x[[index]])
    aux_bhat[k] = np.sin(xbar_aux[k])
  
  beta_b = np.mean(aux_bhat)
  bias_corr_b[i] = 2 * np.sin(np.mean(x)) - beta_b
  
  xbar_b[i] = np.mean(x)
  se_b[i] = np.sqrt(np.cos(xbar_b[i]) * np.var(x) / n * (np.cos(xbar_b[i])))
  trat_b[i] = (bias_corr_b[i] - beta) / se_b[i]
  lcl_asym_b[i] = bias_corr_b[i] - 1.96 * se_b[i]
  ucl_asym_b[i] = bias_corr_b[i] + 1.96 * se_b[i]
  
  # jackknife
  xbar_aux = np.zeros((n, 1))
  aux_bhat = np.zeros((n, 1))
  for k in range(n):
    ind = np.ones(n, bool)
    ind[k] = False
    xbar_aux[k] = np.mean(x[ind])
    aux_bhat[k] = np.sin(xbar_aux[k])
    
  beta_jk = np.mean(aux_bhat)
  bias_corr_jk[i] = n * np.sin(np.mean(x)) - (n - 1) * beta_jk

  trat_jk[i] = (bias_corr_jk[i] - beta) / se_b[i]
  lcl_asym_jk[i] = bias_corr_jk[i] - 1.96 * se_b[i]
  ucl_asym_jk[i] = bias_corr_jk[i] + 1.96 * se_b[i]


cov_freq_asym_b = np.mean(np.logical_and(beta > lcl_asym_b, beta < ucl_asym_b))
cov_freq_asym_jk = np.mean(np.logical_and(beta > lcl_asym_jk, beta < ucl_asym_jk))
  

#%%
 

cov_freq_asym_b = np.mean(np.logical_and(beta > lcl_asym_b, beta < ucl_asym_b))









  
  
  
  

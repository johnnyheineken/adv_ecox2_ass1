# %  Name                   	Student id                email
# % +------------------------+------------------------+-------------------------
# % |       Jan Hynek        |      11748494          | janhynek@gmail.com
# % +------------------------+------------------------+-------------------------
# % |     Stepan Svoboda     |                        |
# % +------------------------+------------------------+-------------------------
# % I (enlisted above) declare that:
# %   1. Our assignment will be our own work.
# %   2. We shall not make solutions to the assignment available to anyone else.
# %   3. We shall not engage in any other activities that will dishonestly improve my results or dishonestly improve or hurt the results of others.

#%%
import numpy as np
from numpy import sin, cos, mean, exp, mean, sqrt
import pandas as pd
np.random.seed(2110)
#%%
n = 50
print(n)
REP = 1000
print(REP)
BOOTREP = 499
print(BOOTREP)
m = 0
s = 1.3

mu = exp(m + (1/2 * s ** 2))
print(mu)
beta = sin(mu)
print(beta)
#%%
# % average original sample
xbar = np.zeros(REP)
# % estimate of beta
bhat = np.zeros(REP)
# % estimate of beta
bhat_jk = np.zeros(REP)
# % estimate of beta
bhat_bs = np.zeros(REP)
# % standard error bhat(asymptotic)
SE = np.zeros(REP)
SE_bs = np.zeros(REP)
SE_jk = np.zeros(REP)
# % t - ratio
trat = np.zeros(REP)
# % Lower confidence limit(asym)
LCLasym = np.zeros(REP)
# % Upper confidence limit(asym)
UCLasym = np.zeros(REP)
# % Lower confidence limit(asym)
LCLasym_jk = np.zeros(REP)
# % Upper confidence limit(asym)
UCLasym_jk = np.zeros(REP)
# % Lower confidence limit(asym)
LCLasym_bs = np.zeros(REP)
# % Upper confidence limit(asym)
UCLasym_bs = np.zeros(REP)
#%%
n = 100
for i in range(REP):
    if i%100 == 0: 
        print(i)
    X = exp(np.random.normal(loc=m, scale=s, size = n))
    xbar[i] = mean(X)
    bhat[i] = sin(xbar[i])
    SE[i] = np.sqrt(np.var(X) * 1/n) * np.abs(cos(xbar[i]))
    trat[i] = (bhat[i] - beta) / SE[i]
    LCLasym[i] = bhat[i] - 1.96 * SE[i]
    UCLasym[i] = bhat[i] + 1.96 * SE[i]
print(n)
CoverageFreqasym = mean((beta > LCLasym) & (beta < UCLasym))
print('Coverage freq.(asym):'   + str(CoverageFreqasym))
#%%
CoverageFreqasym = 0
n = 50
while (CoverageFreqasym < 0.9365) or (CoverageFreqasym > 0.9635):
    for i in range(REP):
        if i % 100 == 0:
            print(i)
        X = exp(np.random.normal(loc=m, scale=s, size=n))
        xbar[i] = mean(X)
        bhat[i] = sin(xbar[i])
        SE[i] = np.sqrt(np.var(X) * 1 / n) * np.abs(cos(xbar[i]))
        trat[i] = (bhat[i] - beta) / SE[i]
        LCLasym[i] = bhat[i] - 1.96 * SE[i]
        UCLasym[i] = bhat[i] + 1.96 * SE[i]
    CoverageFreqasym = mean((beta > LCLasym) & (beta < UCLasym))
    n += 50
print(n - 50)


#%%
def calculate_jk(X):
    N = len(X)
    mu_jk = np.zeros(N)
    for i in range(N):
        jk_X = X[:i] + X[(i + 1):]
        mu_X_jk = mean(jk_X)
        mu_jk[i] = mu_X_jk
    theta_jk = mean(mu_jk)
    theta_jk_BC = N * sin(mean(X)) - (N - 1) * sin(theta_jk)
    se_jk = sqrt(N - 1 / N * sum([(i - theta_jk_BC) ** 2 for i in mu_jk]))
    return theta_jk_BC, se_jk



#%%
def calculate_bs(X, bootrep):
    N = len(X)
    mu_bs = np.zeros(bootrep)
    for i in range(bootrep):
        indices = np.random.randint(0, high = N, size = N)
        bootstrap = [X[j] for j in indices.tolist()]
        mu_bs[i] = mean(bootstrap)
    theta_bs = 2 * sin(mean(X)) - sin(mean(mu_bs))
    se_bs = (1/(bootrep - 1)) * sum([(i-theta_bs)**2 for i in mu_bs])
    return theta_bs, se_bs

#%%
n = 50
for i in range(REP):
    if i % 100 == 0:
        print(i)
    X = exp(np.random.normal(loc=m, scale=s, size=n))
    bhat_jk[i], SE_jk[i] = calculate_jk(X.tolist())
    bhat_bs[i], SE_bs[i] = calculate_bs(X.tolist(), BOOTREP)
    xbar[i] = mean(X)
    bhat[i] = sin(xbar[i])
    SE[i] = np.sqrt(np.var(X) * 1 / n) * np.abs(cos(xbar[i]))
    trat[i] = (bhat[i] - beta) / SE[i]
    LCLasym[i] = bhat[i] - 1.96 * SE[i]
    UCLasym[i] = bhat[i] + 1.96 * SE[i]
    LCLasym_jk[i] = bhat_jk[i] - 1.96 * SE[i]
    UCLasym_jk[i] = bhat_jk[i] + 1.96 * SE[i]
    LCLasym_bs[i] = bhat_bs[i] - 1.96 * SE[i]
    UCLasym_bs[i] = bhat_bs[i] + 1.96 * SE[i]
CoverageFreqasym = mean((beta > LCLasym) & (beta < UCLasym))
CoverageFreqasym_jk = mean((beta > LCLasym_jk) & (beta < UCLasym_jk))
CoverageFreqasym_bs = mean((beta > LCLasym_bs) & (beta < UCLasym_bs))

print(CoverageFreqasym)
print(CoverageFreqasym_bs)
print(CoverageFreqasym_jk)


#%%
n = 50
for i in range(REP):
    if i % 100 == 0:
        print(i)
    X = exp(np.random.normal(loc=m, scale=s, size=n))
    bhat_jk[i], SE_jk[i] = calculate_jk(X.tolist())
    bhat_bs[i], SE_bs[i] = calculate_bs(X.tolist(), BOOTREP)
    xbar[i] = mean(X)
    bhat[i] = sin(xbar[i])
    SE[i] = np.sqrt(np.var(X) * 1 / n) * np.abs(cos(xbar[i]))
    trat[i] = (bhat[i] - beta) / SE[i]
    LCLasym[i] = bhat[i] - 1.96 * SE[i]
    UCLasym[i] = bhat[i] + 1.96 * SE[i]
    LCLasym_jk[i] = bhat[i] - 1.96 * SE_jk[i]
    UCLasym_jk[i] = bhat[i] + 1.96 * SE_jk[i]
    LCLasym_bs[i] = bhat[i] - 1.96 * SE_bs[i]
    UCLasym_bs[i] = bhat[i] + 1.96 * SE_bs[i]
CoverageFreqasym = mean((beta > LCLasym) & (beta < UCLasym))
CoverageFreqasym_jk = mean((beta > LCLasym_jk) & (beta < UCLasym_jk))
CoverageFreqasym_bs = mean((beta > LCLasym_bs) & (beta < UCLasym_bs))

print(CoverageFreqasym)
print(CoverageFreqasym_bs)
print(CoverageFreqasym_jk)

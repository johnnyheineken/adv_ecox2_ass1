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
from numpy import sin, cos, mean, exp, mean, sqrt, percentile, std, var
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

bhat = exp(m + (1/2 * s ** 2))
print(bhat)
beta = sin(bhat)
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
def calculate_orig(X):
    xbar = mean(X)
    bhat = sin(xbar)
    SE = np.sqrt(np.var(X) * 1 / n) * np.abs(cos(xbar))
    return mean(X), bhat, SE

def calculate_jk(X):
    N = len(X)
    bhat_jk = np.zeros(N)
    for i in range(N):
        jk_X = X[:i] + X[(i + 1):]
        bhat_X_jk = mean(jk_X)
        bhat_jk[i] =sin(bhat_X_jk)
    theta_jk = mean(bhat_jk)
    theta_jk_BC = N * sin(mean(X)) - (N - 1) * theta_jk
    se_jk = sqrt((N - 1) * np.var(bhat_jk))
    return theta_jk_BC, se_jk


def calculate_bs(X, bootrep):
    N = len(X)
    bhat_bs = np.zeros(bootrep)
    for i in range(bootrep):
        indices = np.random.randint(0, high = N, size = N)
        bootstrap = [X[j] for j in indices.tolist()]
        bhat_bs[i] = sin(mean(bootstrap))
    theta_bs = 2 * sin(mean(X)) - mean(bhat_bs)
    se_bs = np.std(bhat_bs)
    return theta_bs, se_bs

#%%
n = 50
report_numbers = FALSE
for i in range(REP):
    if (i % 100) == 0 and report_numbers:
        print(i)
    X = exp(np.random.normal(loc=m, scale=s, size=n))
    bhat_jk[i], SE_jk[i] = calculate_jk(X.tolist())
    bhat_bs[i], SE_bs[i] = calculate_bs(X.tolist(), BOOTREP)
    xbar[i], bhat[i], SE[i] = calculate_orig(X.tolist())
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

print('orig: ' + str(CoverageFreqasym))
print('BS: ' + str(CoverageFreqasym_bs))
print('JK: ' + str(CoverageFreqasym_jk))


#%%



def calculate_bs2(X, bootrep):
    N = len(X)
    bhat_bs = np.zeros(bootrep)
    tstat = np.zeros(bootrep)
    bhat = sin(mean(X))
    for i in range(bootrep):
        indices = np.random.randint(0, high=N, size=N)
        bootstrap = [X[j] for j in indices.tolist()]
        bhat_bs[i] = sin(mean(bootstrap))
        # se = sqrt(var(bootstrap) * 1 / N) * np.abs(cos(bhat_bs[i]))
        # tstat = (bhat_bs[i] - bhat) / se

    se_bs = std(bhat_bs)
    tstat = [(i - beta) / se_bs for i in bhat_bs.tolist()]
    theta_bs = 2 * bhat - mean(bhat_bs)
    lower_perc = np.percentile(bhat_bs, 2.5)
    upper_perc = np.percentile(bhat_bs, 97.5)
    lower_tstat = np.percentile(tstat, 2.5)
    upper_tstat = np.percentile(tstat, 97.5)
    return theta_bs, se_bs, lower_perc, upper_perc, lower_tstat, upper_tstat


lower_perc = np.zeros(REP)
upper_perc= np.zeros(REP)
lower_tstat = np.zeros(REP)
upper_tstat = np.zeros(REP)


n = 50
report_numbers = True
for i in range(REP):
    if (i % 100) == 0 and report_numbers:
        print(i)
    X = exp(np.random.normal(loc=m, scale=s, size=n))
    bhat_jk[i], SE_jk[i] = calculate_jk(X.tolist())
    (bhat_bs[i], SE_bs[i],
     lower_perc[i], upper_perc[i],
     lower_tstat[i], upper_tstat[i]) = calculate_bs2(X.tolist(), BOOTREP)
    xbar[i], bhat[i], SE[i] = calculate_orig(X.tolist())
    trat[i] = (bhat[i] - beta) / SE[i]
    LCLasym[i] = lower_perc[i]
    UCLasym[i] = upper_perc[i]
    LCLasym_jk[i] = bhat[i] - upper_tstat[i] * SE[i]
    UCLasym_jk[i] = bhat[i] - lower_tstat[i] * SE[i]
    # LCLasym_bs[i] = bhat[i] - 1.96 * SE_bs[i]
    # UCLasym_bs[i] = bhat[i] + 1.96 * SE_bs[i]
    

percentile_method = mean((beta > LCLasym) & (beta < UCLasym))
percentile_t_method = mean((beta > LCLasym_jk) & (beta < UCLasym_jk))
# CoverageFreqasym_bs = mean((beta > LCLasym_bs) & (beta < UCLasym_bs))

print('percentile: ' + str(percentile_method))
# print('BS: ' + str(CoverageFreqasym_bs))
print('percentile-t: ' + str(percentile_t_method))


#%%
print(mean(bhat) - mean(lower_tstat) * mean(SE))
print(mean(bhat) - mean(upper_tstat) * mean(SE))
print(mean(bhat))
print(mean(LCLasym_jk))
print(mean(UCLasym_jk))
print(mean(LCLasym))
print(mean(UCLasym))
#%%
print(beta)

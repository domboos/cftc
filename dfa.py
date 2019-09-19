# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:25:47 2019

@author: grbi
"""

# IMPORT DATA:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# File directory: change paths
os.getcwd()
os.chdir('C:\\Users\\bood\\Documents')
# os.getcwd()

nonC = pd.read_excel('Net_NonC.xlsx')

fut = pd.read_excel('Prices.xlsx')
price = fut['CL1']


def momlog(price, lag):
    momRet = pd.DataFrame({'Dates': fut['Dates']})
    for i in lag:
        momRet[('mom' + str(i))] = np.log(price / price.shift(i))
    return momRet


# the smaller the differences between the momentum signal, the higher is the correlation in between.
lbps = [5, 10] + list(range(21, 85, 21)) + list(range(85, 300, 42))
print(lbps)
# lbps = range(1,260,1)


# calculate momentum
mom = momlog(price, lbps).dropna().reset_index(drop=True)

# corrleation between the different momentum signals
# import seaborn as sns
# cormat = mom.corr()
# plt.figure()
# sns.set(font_scale=0.9)
# g = sns.heatmap(cormat, vmin=0, vmax=1,yticklabels=True,xticklabels=True)
# g.set_yticklabels(g.get_yticklabels(), fontsize = 10)
# plt.show()

# Postition of CFTC traders: y
pos = pd.DataFrame({'Dates': nonC['Dates']})
pos['pos'] = nonC['CL1']
pos = pos[(pos['Dates'] > mom['Dates'].iloc[0])]  # match dates

df = pd.merge(pos, mom.shift(1), how='left', on='Dates')
df.drop(df.index[-1], inplace=True)  # drop last row

# def X and y for ML algos.
Xs = df.iloc[:, 2:]
y = df.iloc[:, 1]

# -----------------------------------------------------------------------------
# Lasso:
from sklearn.linear_model import (LinearRegression, Ridge, Lasso)

# alpha = 0     --> obj is OLS, estimation as in OLS
# alpha = inf   --> coef = 0, bc infinite weightage given on square coefs.
# 0 <alpha < inf-->  The magnitude of α will decide the weightage given to different parts of objective. The coefficients will be somewhere between 0 and ones for simple linear regression.

lasso = Lasso(alpha=10 ** -4, normalize=True).fit(Xs, y)  # lasso alpha = 10**-4 or smaller
lasso.fit_intercept
lasso.coef_
print(lasso.coef_)
print(lasso.score(Xs, y))

# cross validation approach
# Import the necessary module
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, Xs, y, cv=5)

# Print the 5-fold cross-validation scores
print('5-fold cross-validation scores')
print(cv_scores)

# find the mean of our cv scores here
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

colnames = Xs.columns

# Plot that shows, which factors are the most important ones:
plt.plot(range(len(colnames)), lasso.coef_)
plt.xticks(range(len(colnames)), colnames.values, rotation=60)
# -----------------------------------------------------------------------------


# Principle Component Analysis:
from sklearn.decomposition import PCA

num_pc = 2

portfolio_returns = Xs

X = np.asarray(portfolio_returns)
[n, m] = X.shape
print('The number of timestamps is {}.'.format(n))
print('The number of stocks is {}.'.format(m))

pca = PCA(n_components=num_pc)  # number of principal components
pca.fit(X)

percentage = pca.explained_variance_ratio_
percentage_cum = np.cumsum(percentage)
print('{0:.2f}% of the variance is explained by the first 2 PCs'.format(percentage_cum[-1] * 100))

pca_components = pca.components_

# contribution of the first two components:
x = np.arange(1, len(percentage) + 1, 1)

plt.subplot(1, 2, 1)
plt.bar(x, percentage * 100, align="center")
plt.title('Contribution of principal components', fontsize=16)
plt.xlabel('principal components', fontsize=16)
plt.ylabel('percentage', fontsize=16)
plt.xticks(x, fontsize=16)
plt.yticks(fontsize=16)
plt.xlim([0, num_pc + 1])

plt.subplot(1, 2, 2)
plt.plot(x, percentage_cum * 100, 'ro-')
plt.xlabel('principal components', fontsize=16)
plt.ylabel('percentage', fontsize=16)
plt.title('Cumulative contribution of principal components', fontsize=16)
plt.xticks(x, fontsize=16)
plt.yticks(fontsize=16)
plt.xlim([1, num_pc])
plt.ylim([50, 100]);

# how much of the momentum signals comes from statistical noise:
factor_returns = X.dot(pca_components.T)
factor_returns = pd.DataFrame(columns=["factor 1", "factor 2"],
                              index=portfolio_returns.index,
                              data=factor_returns)
factor_returns.head()

factor_exposures = pd.DataFrame(index=["factor 1", "factor 2"],
                                columns=portfolio_returns.columns,
                                data=pca.components_).T
factor_exposures

colnames = Xs.columns
plt.figure()
plt.plot(range(len(colnames)), factor_exposures['factor 1'])
plt.plot(range(len(colnames)), factor_exposures['factor 2'])
plt.xticks(range(len(colnames)), colnames.values, rotation=60)

labels = factor_exposures.index
data = factor_exposures.values
plt.subplots_adjust(bottom=0.1)
plt.scatter(
    data[:, 0], data[:, 1], marker='o', s=300, c='m',
    cmap=plt.get_cmap('Spectral'))
plt.title('Scatter Plot of Coefficients of PC1 and PC2')
plt.xlabel('factor exposure of PC1')
plt.ylabel('factor exposure of PC2')

for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    );


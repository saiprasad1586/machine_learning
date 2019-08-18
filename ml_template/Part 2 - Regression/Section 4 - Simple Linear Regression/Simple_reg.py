# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 00:47:07 2019

@author: GVND SAIPRASAD
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data
dataset = pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,1].values


#Splitting the dataset into trainning and testing
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting Simple Linear Regression to the traininig Set

from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the test set results
y_pred=regressor.predict(X_test)

# visualising the training set results
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience(Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

# visualising the test set results
plt.scatter(X_test,Y_test,color="green")
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience(Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

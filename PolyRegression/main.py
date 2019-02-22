import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import sklearn.

df = pd.read_table('dataset1_train.txt',delim_whitespace=True, names=('X1','X2','Y'))
print(df)

X = df[['X1','X2']];
target = df[['Y']];

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,target, test_size = 0.25, random_state = 0)
print(X_test,"\n",y_test)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

lin = LinearRegression()
lin.fit(X_train,y_train)
y_pred = lin.predict(X_test)
print(y_pred,"\n",y_test)
accuracy1 = lin.score(X_test,y_test)


poly = PolynomialFeatures(degree=4)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.fit_transform(X_test)
poly.fit(X_train, y_train)
lin2 = LinearRegression()
lin2.fit(X_poly_train,y_train);

y_pred2 = lin2.predict(X_poly_test)
print(y_pred2,"\n",y_test)
accuracy2 = lin2.score(X_poly_test,y_test)
print("the accuracy is :: 1 ==>",accuracy1*100," and 2 ==>",accuracy2*100,'%')

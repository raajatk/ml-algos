# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('dataset.csv')

target = df['county']

print(df)
df=df.drop(['zip_code','city','state','county'],axis=1)
print("The Dataframe is \n",df)
print("The target is \n",target)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df,target, test_size = 0.25, random_state = 0)
print(X_test,"\n",y_test)
#Finding the Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#

# for n in range(1,15):
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

print("the test data is \n",y_test,"\n","the predicted data is \n",y_pred,"\nthe accuracy is ",accuracy)
# plt.figure(figsize=(16,9))
# plt.plot(range(1,15),accuracy)
# plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

# print(iris)
print("The X is \n",X);
print("The target is \n",y);

tree = RandomForestClassifier(n_estimators = 100, random_state = 42);

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

print(X_train,y_test)

model = tree.fit(X_train,y_train)

model.predict(X_test)

predicted = model.predict(X_test);
print(predicted)
print(accuracy_score(y_test,predicted)*100,"%")

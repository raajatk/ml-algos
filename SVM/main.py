from sklearn import svm, datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

# print(iris)
print("The X is \n",X);
print("The target is \n",y);

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

print(X_train,y_test)

model = svm.SVC(kernel='linear' ,C=1,gamma=1);

model.fit(X_train,y_train);
model.score(X_train,y_train);

predicted = model.predict(X_test);
print(accuracy_score(y_test,predicted)*100,"%")

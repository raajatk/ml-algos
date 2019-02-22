from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target

# print(iris)
print("The X is \n",X);
print("The target is \n",y);

tree = DecisionTreeClassifier()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

print(X_train,y_test)

model = tree.fit(X_train,y_train)

model.predict(X_test)

predicted = model.predict(X_test);

from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin'

dot_data = StringIO()

export_graphviz(
    model,
    out_file = dot_data,
    filled=True, rounded=True, proportion=False,
    special_characters=True,
    feature_names=['1','2','3','4'],
    class_names=["0", "1",'2']
)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
graph.write_png('tree.png')
print(accuracy_score(y_test,predicted)*100,"%")

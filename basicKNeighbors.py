from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.5)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()

classifier = classifier.fit(x_train,y_train)

predictions = classifier.predict(x_test)

from sklearn.metrics import accuracy_score

print accuracy_score(y_test,predictions) #gives the accuracy between test target set and predictions. 100% is the optimum.

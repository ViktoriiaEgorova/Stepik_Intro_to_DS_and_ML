from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_data = pd.read_csv('train.csv')
# print(titanic_data.head())
#print(titanic_data.isnull())

X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data.Survived
# конвертируем строковые данные в числовые
X = pd.get_dummies(X)
# заполняем пропущенные значения
X = X.fillna({'Age' : X.Age.median()})

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(clf.score(X, y))

clf.fit(X_train, y_train)
print(clf.score(X, y))
print(clf.score(X_test, y_test))

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
clf.fit(X_test, y_test)
print(clf.score(X_test, y_test))



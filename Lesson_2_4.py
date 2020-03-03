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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#######################################################################################################################

max_depth_values = range(1, 100)
scores_data = pd.DataFrame()
for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    clf.fit(X_test, y_test)
    test_score = clf.score(X_test, y_test)
    temp_score_data = pd.DataFrame({'max_depth':[max_depth], 'train_score':[train_score], 'test_score':[test_score]})
    scores_data = scores_data.append(temp_score_data)

#print(scores_data.head())

scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score', 'test_score'], var_name= 'set_type', value_name='score')
#print(scores_data_long.head())

#sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)
#plt.show()

#######################################################################################################################

from sklearn.model_selection import cross_val_score
# clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
# print(cross_val_score(clf, X_train, y_train, cv=5).mean())
# cross_val_score(clf, X_train, y_train, cv=5).mean()

#######################################################################################################################

max_depth_values = range(1, 100)
scores_data = pd.DataFrame()
for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    clf.fit(X_test, y_test)
    test_score = clf.score(X_test, y_test)
    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    temp_score_data = pd.DataFrame({'max_depth':[max_depth], 'train_score':[train_score], 'test_score':[test_score], 'cross_val_score': [mean_cross_val_score]})
    scores_data = scores_data.append(temp_score_data)

print(scores_data.head())

scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score', 'test_score', 'cross_val_score'], var_name= 'set_type', value_name='score')
print(scores_data_long.head())

sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)
plt.show()
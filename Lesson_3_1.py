from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

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

# clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=200)
# clf.fit(X_train, y_train)
#
# tree.plot_tree(clf.fit(X, y))
# plt.show()

from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
parameters = {'n_estimators':[10, 20, 30], 'max_depth':[2, 5, 7, 10]}
grid_search_cv_clf = GridSearchCV(clf_rf, parameters, cv=5)
grid_search_cv_clf.fit(X_train, y_train)
grid_search_cv_clf.best_params_

# rf = RandomForestClassifier(n_estimators=15, max_depth=5)
# #parameters = {'n_estimators':15, 'max_depth':5}
# #rf = grid_search_cv_clf = GridSearchCV(clf_rf, cv=5)
# rf.fit(X_train, y_train)
# predictions = rf.predict(X_test)
# print(predictions)

best_clf = grid_search_cv_clf.best_estimator_
best_clf.score(X_test, y_test)
feature_importances = best_clf.feature_importances_
feature_importances_df = pd.DataFrame({'features':list(X_train), 'feature_importances':feature_importances})
print(feature_importances_df.sort_values('feature_importances', ascending=False))
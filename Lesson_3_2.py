from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

########################################################################################################################

heart_data = pd.read_csv('heart-disease-uci.zip', compression ='zip')

print(heart_data.head())

# X = heart_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
# y = heart_data.Survived
# # конвертируем строковые данные в числовые
# X = pd.get_dummies(X)
# # заполняем пропущенные значения
# X = X.fillna({'Age' : X.Age.median()})
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

########################################################################################################################

from sklearn.ensemble import RandomForestClassifier

np.random.seed(0)

rf = RandomForestClassifier(10, max_depth=5)
#rf.fit(X_train, y_train)

# best_clf = grid_search_cv_clf.best_estimator_
# best_clf.score(X_test, y_test)
# feature_importances = best_clf.feature_importances_
# feature_importances_df = pd.DataFrame({'features':list(X_train), 'feature_importances':feature_importances})
# print(feature_importances_df.sort_values('feature_importances', ascending=False))
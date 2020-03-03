from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#data = pd.read_csv('example_data.csv')
data = pd.DataFrame({'X_1': [1, 1, 1, 0, 0, 0, 0, 1], 'X_2': [0, 0, 0, 1, 0, 0, 0, 1], 'Y': [1, 1, 1, 1, 0, 0, 0, 0]})

clf = tree.DecisionTreeClassifier(criterion='entropy')

X = data[['X_1', 'X_2']]
y = data.Y

clf.fit(X, y)

tree.plot_tree(clf.fit(X, y))
plt.show()

import math


E_sh_sob = -(1)*math.log2(1)
E_sh_kot = -(4/9)*math.log2(4/9) - (5/9)*math.log2(5/9)

E_gav_sob = -(1)*math.log2(1)
E_gav_kot = -(4/5)*math.log2(4/5) - (1/5)*math.log2(1/5)

E_laz_sob = -(1)*math.log2(1)
E_laz_kot = -(1)*math.log2(1)

print(E_sh_sob, E_sh_kot)
print(E_gav_sob, E_gav_kot)
print(E_laz_sob, E_laz_kot)
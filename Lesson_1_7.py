import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

students_performance = pd.read_csv('StudentsPerformance.csv')
students_performance.rename(columns={'math score' : 'math_score'} )
students_performance.columns = students_performance.columns.str.replace(' ', '_')
print(list(students_performance))
#students_performance.math_score.hist()
#students_performance.plot.scatter(x='math_score', y='reading_score')
#plt.show()
#ax = sns.lmplot(x='math_score', y='reading_score', hue='gender', data=students_performance)
#ax.set_xlabels('math score')
#plt.show()

# df = pd.read_csv('income.csv')
# plt.plot(df.index, df.income)
# plt.show()

df = pd.read_csv("dataset_209770_6.txt", sep=" ")
#ax = sns.lmplot(x='x', y='y', data=df)
#plt.show()

#df = pd.read_csv('genome_matrix.csv')
df = pd.read_csv('genome_matrix.csv', index_col=0)
#print(df)
#sns.heatmap(g, cmap='viridis')
# g = sns.heatmap(df, cmap='viridis')
# g.xaxis.set_ticks_position('top')
# g.xaxis.set_tick_params(rotation=90)
#plt.show()

dota = pd.read_csv('dota_hero_stats.csv')
#print(dota['roles'].mode())

iris = pd.read_csv('iris.csv')
#new_iris = iris.iloc[:,2:6]
# for column in iris:
#     sns.distplot(iris[column])
#     plt.show()
#sns.violinplot(iris['petal length'])
#print(iris.iloc[:,1:5])
sns.pairplot(iris.iloc[:,1:5])
plt.show()
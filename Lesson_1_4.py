import pandas as pd
import numpy as np

students_performance = pd.read_csv('StudentsPerformance.csv')

#первые 5 строк (не включая 5ую), первые 3 столбца
students_performance.iloc[0:5, 0:3]

students_performance_with_names = students_performance.iloc[[0, 3, 4, 7, 8]]
students_performance_with_names.index = ['A', 'B', 'C', 'D', 'E']

titanic = pd.read_csv('titanic.csv')
print(type(titanic))
print(titanic.shape)
print(titanic.dtypes.value_counts())
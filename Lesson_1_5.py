import pandas as pd
import numpy as np

students_performance = pd.read_csv('StudentsPerformance.csv')

free = students_performance.loc[students_performance.lunch == 'free/reduced']
print(free.shape[0] / students_performance.shape[0])

standard = students_performance.loc[students_performance.lunch == 'standard']

print(free.describe())
print(standard.describe())


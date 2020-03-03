import pandas as pd
import numpy as np

s1 = pd.Series(['A', 'A', 'B' , 'B'])
s2 = pd.Series([10, 14, 12 , 23])
my_data = pd.DataFrame({'type':s1, 'value': s2})
#print(my_data)

my_stat = pd.read_csv('my_stat.csv')
subset_1 = my_stat[(my_stat.V1>0) & (my_stat.V3 == 'A')]
subset_2 = my_stat[(my_stat.V4>=1) | (my_stat.V2 != 10)]

my_stat['V5'] = my_stat.V1 + my_stat.V4
my_stat['V6'] = np.log(my_stat.V2)

my_stat = my_stat.rename(columns={'V1':'session_value', 'V2':'group', 'V3':'time', 'V4':'n_users'})

my_stat['session_value'].fillna(0)
my_stat_pos = my_stat[my_stat.n_users>0]
med = my_stat_pos.median()
my_stat.loc[my_stat['n_users'] <0, 'n_users'] = med
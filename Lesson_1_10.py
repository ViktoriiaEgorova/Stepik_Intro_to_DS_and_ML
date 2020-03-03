import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

events_data = pd.read_csv('event_data_train.zip', compression ='zip')

# print(events_data.head())
# print(events_data.action.unique())

events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')

events_data['day'] = events_data.date.dt.date
print(events_data.head())
#print(events_data.groupby('day').user_id.nunique().head(20))
#events_data.groupby('day').user_id.nunique().plot()
#plt.show()

# посчитаем число пользователей прошедших степы для всех пользователей

p_table = events_data.pivot_table(index='user_id', columns='action', values='step_id', aggfunc='count', fill_value=0).head()
#p_table.hist()
#plt.show()

submissions_data = pd.read_csv('submissions_data_train.zip', compression ='zip')
user_scores = submissions_data.pivot_table(index='user_id', columns='submission_status', values='step_id', aggfunc='count', fill_value=0).reset_index()
#print(user_scores.head())

t = events_data[['user_id', 'day', 'timestamp']].drop_duplicates(subset=['user_id', 'day']).groupby('user_id')['timestamp'].apply(list)

gap_data = t.apply(np.diff).values
gap_data = pd.Series(np.concatenate(gap_data,axis=0))
gap_data = gap_data / (24*60*60)

# print(gap_data)

#print(submissions_data)


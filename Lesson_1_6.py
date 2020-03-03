import pandas as pd
import numpy as np

dota = pd.read_csv('dota_hero_stats.csv')

#print(list(dota))
#print(dota.head(2))

column_legs = dota.groupby('legs')

#print(dota.loc[dota.legs == 0])
#print(dota.loc[dota.legs == 8].shape)


accountancy = pd.read_csv('accountancy.csv')
#print(accountancy)

#print(accountancy.groupby(['Executor','Type']).aggregate({'Salary':'mean'}))

#print(dota.groupby(['attack_type', 'primary_attr']).nunique())

algae = pd.read_csv('algae.csv')

#print(list(algae))

#concentrations = algae.groupby(['genus', 'oleic_acid']).aggregate({'oleic_acid':'mean'})
concentrations = algae.groupby(['genus']).mean()
#concentrations = algae.groupby(['genus', 'oleic_acid']).aggregate({'oleic_acid':'mean'})
#print(concentrations)

c1 = algae.loc[algae.genus == 'Fucus']
c = c1['alanin']
#print(c.min(), c.mean(), c.max())

print(list(algae))
print(algae.groupby('group').describe())
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as sts
import functions

data = pd.read_csv('mall_data')
#print(data.head())
data.columns = ['id', 'Gender', 'age', 'income', 'score']

male_income = data[data['Gender'] == 'Male'].income.values.tolist()
male_score = data[data['Gender'] == 'Male'].score.values.tolist()
female_score = data[data['Gender'] == 'Female'].score.values.tolist()
female_income = data[data['Gender'] == 'Female'].income.values.tolist()

#functions.clusters(male_income, male_score, ((25, 20), (25, 70), (55, 50), (85, 15), (85, 80)))

#functions.clusters(female_income, female_score, ((25, 20), (25, 80), (55, 50), (90, 20), (90, 80)))

print(functions.elem_stats(male_income),
functions.elem_stats(male_score),
functions.elem_stats(female_income),
functions.elem_stats(female_score))

m_a = data[data['Gender'] == 'Male'].age.values.tolist()
f_a = data[data['Gender'] == 'Female'].age.values.tolist()

a = functions.correlation(male_income, m_a)
b = functions.correlation(female_income, f_a)
c = functions.correlation(m_a, male_score)
d = functions.correlation(f_a, female_score)
print(a, b, c, d)


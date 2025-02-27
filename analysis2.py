# prvi podaci koje koristimo su SSA odnosno površina sunčevih pega od maja 1874 do aprila 2021

import pandas as pd

ssa_path = '/home/bogdan/Desktop/sunspot_analysis/data/sunspot_area.txt'

ssa = pd.read_csv(ssa_path, skiprows=1, delim_whitespace=True, header=None, names=['year', 'month', 'area'])

# spajamo goodinu i mesec u jednu kolonu datum i kovertujemo u format datuma
# izbacujemo year i month kolone
ssa['date'] = pd.to_datetime(ssa[['year', 'month']].assign(day=1))
ssa = ssa.drop(columns=['year', 'month'])

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Vizuelizacija vremenske serije
plt.figure(figsize=(12, 6))
plt.plot(ssa['date'], ssa['area'], label='SSA')
plt.title('Površina Sunčevih pega (SSA) tokom vremena')
plt.xlabel('Datum')
plt.ylabel('SSA (µHem)')
plt.legend()
plt.grid(True)
plt.show()

# 2. Provera periodičnosti (autokorelacija)
from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(12, 6))
autocorrelation_plot(ssa['area'])
plt.title('Autokorelacija SSA podataka')
plt.show()

# 3. Provera stacionarnosti 
# konstantna srednja vrednost podataka kroz vreme
# konstantna varijansa (rasuta vrednost) podataka kroz vreme oko srednje vrednosti
# konstantna autokorelacija podataka

from statsmodels.tsa.stattools import adfuller

result = adfuller(ssa['area'])

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))





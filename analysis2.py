# prvi podaci koje koristimo su SSA odnosno površina sunčevih pega od maja 1874 do aprila 2021

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ssa_path = '/home/bogdan/Desktop/sunspot_analysis/data/sunspot_area.txt'

ssa = pd.read_csv(ssa_path, skiprows=1, sep='\s+', header=None, names=['year', 'month', 'area'])
ssa['area'] = ssa['area'].replace(0.0, np.nan).interpolate()

# spajamo goodinu i mesec u jednu kolonu datum i kovertujemo u format datuma
# izbacujemo year i month kolone
ssa['date'] = pd.to_datetime(ssa[['year', 'month']].assign(day=1))
ssa = ssa.drop(columns=['year', 'month'])

"""
# za početak koristimo autokorelogram (plot_acf - autocorellation function)
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(ssa['area'])
plt.show()

# zatim radimo parcialni autokorelogram (plot_pacf - partial autocorellation function)
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(ssa['area'])
plt.show()
"""

# grafik liči na multiplikativni model, pa ćemo pokušati da ga transformišemo logaritmovanjem

ssa['log_area'] = np.log(ssa['area'])   

# sada smo dobili aditivni model podataka
# koristimo STL da vremensku seriju razbijemo na komponente
# plotujemo grafik

from statsmodels.tsa.seasonal import STL

"""
stl = STL(ssa['log_area'], seasonal=13, period=132).fit()
stl.plot()
plt.show()
"""

# radimo ADF test da proverimo da li je serija stacionarna
from statsmodels.tsa.stattools import adfuller

"""
adf_value = adfuller(ssa['area'])[0]
p_value = adfuller(ssa['area'])[1]
print(f'ADF value: {adf_value}')
print(f'p value: {p_value}')

if p_value < 0.05:                        # dobijamo jako mali p_value, što znači da je serija stacionarna
    print('Serija je stacionarna')
else:
    print('Serija nije stacionarna')
"""

# podatke imamo zaključno sa januarom 2025, a koristićemo podatke do decembra 2021. godine


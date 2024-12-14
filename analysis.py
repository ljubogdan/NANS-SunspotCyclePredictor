import pandas as pd
import matplotlib.pyplot as plt

file_path = 'data/SN_ms_tot_V2.0.csv'
data = pd.read_csv(file_path, sep=';', header=None, names=['Year', 'Month', 'Decimal_Year', 'Smoothed_Sunspots', 'Error', 'Obs', 'Indicator'])

# izbacimo prvih i zadnjih 6 redova
data = data[6:-6]

# Izbacimo nedostajuće vrednosti
data = data[data['Smoothed_Sunspots'] != -1]

data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Smoothed_Sunspots'], label='Smoothed Sunspots', linewidth=2)
plt.title("13-Month Smoothed Sunspot Numbers", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Sunspot Numbers")
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(data['Smoothed_Sunspots'], model='additive', period=132)

result.plot()
plt.suptitle("Decomposition of Smoothed Sunspot Numbers", fontsize=14)
plt.tight_layout()
plt.show()

from statsmodels.tsa.stattools import adfuller

result = adfuller(data['Smoothed_Sunspots'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(12, 6))
ax1 = plt.subplot(211)
plot_acf(data['Smoothed_Sunspots'], ax=ax1, lags=40)
ax2 = plt.subplot(212)
plot_pacf(data['Smoothed_Sunspots'], ax=ax2, lags=40)
plt.suptitle("ACF and PACF of Smoothed Sunspot Numbers", fontsize=14)
plt.tight_layout()
plt.show()


from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Podesavanje SARIMA modela
sarima_model = SARIMAX(data['Smoothed_Sunspots'], 
                       order=(1, 0, 1), 
                       seasonal_order=(1, 0, 1, 132), 
                       enforce_stationarity=False, 
                       enforce_invertibility=False)

# Treniranje modela
sarima_results = sarima_model.fit()

# Prikaz rezultata
print(sarima_results.summary())

# Vizualizacija predikcije
data['Forecast'] = sarima_results.predict(start=0, end=len(data['Smoothed_Sunspots']) - 1)
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Smoothed_Sunspots'], label='Original Data')
plt.plot(data['Date'], data['Forecast'], label='SARIMA Forecast', color='orange')
plt.title("SARIMA Model Forecast vs Original Data", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Sunspot Numbers")
plt.legend()
plt.tight_layout()
plt.grid(True, alpha=0.5)
plt.show()



















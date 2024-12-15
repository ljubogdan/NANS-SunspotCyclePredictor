import pandas as pd
import matplotlib.pyplot as plt

file_path = 'data/SN_ms_tot_V2.0.csv'
data = pd.read_csv(file_path, sep=';', header=None, names=['Year', 'Month', 'Decimal_Year', 'Smoothed_Sunspots', 'Error', 'Obs', 'Indicator'])

# izbacimo prvih i zadnjih 6 redova
data = data[6:-6]

# Izbacimo nedostajuće vrednosti
data = data[data['Smoothed_Sunspots'] != -1]

data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))

'''
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Smoothed_Sunspots'], label='Smoothed Sunspots', linewidth=2)
plt.title("13-Month Smoothed Sunspot Numbers", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Sunspot Numbers")
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
'''

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(data['Smoothed_Sunspots'], model='additive', period=132)

'''
result.plot()
plt.suptitle("Decomposition of Smoothed Sunspot Numbers", fontsize=14)
plt.tight_layout()
plt.show()
'''

from statsmodels.tsa.stattools import adfuller

'''
result = adfuller(data['Smoothed_Sunspots'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
'''

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


'''
plt.figure(figsize=(12, 6))
ax1 = plt.subplot(211)
plot_acf(data['Smoothed_Sunspots'], ax=ax1, lags=40)
ax2 = plt.subplot(212)
plot_pacf(data['Smoothed_Sunspots'], ax=ax2, lags=40)
plt.suptitle("ACF and PACF of Smoothed Sunspot Numbers", fontsize=14)
plt.tight_layout()
plt.show()
'''


# 80% trening, 20% test
train = data[:int(0.8 * len(data))]
test = data[int(0.8 * len(data)):]


from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

sarima_model = SARIMAX(train['Smoothed_Sunspots'], 
                       order=(1, 0, 1),           
                       seasonal_order=(1, 1, 1, 132), 
                       enforce_stationarity=False, 
                       enforce_invertibility=False)

sarima_model.update([0.9974, 0.9780, 9.495e-05, -0.8152, 4.4278])

# Generisanje predikcija
forecast = sarima_model.filter([0.9974, 0.9780, 9.495e-05, -0.8152, 4.4278]).get_forecast(steps=len(test)).predicted_mean

# Evaluacija
rmse = mean_squared_error(test['Smoothed_Sunspots'], forecast, squared=False)
mae = mean_absolute_error(test['Smoothed_Sunspots'], forecast)

print("RMSE:", rmse)
print("MAE:", mae)

# Vizualizacija predikcija i stvarnih vrednosti
plt.figure(figsize=(10, 5))
plt.plot(test['Date'], test['Smoothed_Sunspots'], label='Stvarne vrednosti', linewidth=2)
plt.plot(test['Date'], forecast, label='Predikcije', color='red', linewidth=2)
plt.legend()
plt.title('Stvarne vrednosti vs Predikcije')
plt.xlabel('Datum')
plt.ylabel('Sunčeve pege')
plt.grid(alpha=0.3)
plt.show()

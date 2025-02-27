import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

file_path = '/kaggle/input/dataset/SN_ms_tot_V2.0.csv'
data = pd.read_csv(file_path, sep=';', header=None, names=['Year', 'Month', 'Decimal_Year', 'Smoothed_Sunspots', 'Error', 'Obs', 'Indicator'])

data = data[6:-6]

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

# cilj projekta je napraviti LSTM+ model koji će predvidjati kako izgleda naredni solarni ciklus
# pošto se poslednji pik desio 2014 godine, probaćemo da predvidimo taj solarni ciklus
# podatke imamo od 1749 godine, tako da ćemo koristiti podatke od 1749 do 2008 godine za treniranje modela
# na istom grafiku plotujemo stvarne podatke i poslednje vrednosti predikcije da se ovelapuju da uporedimo
# kako se model ponaša

train_data = data[data['Year'] <= 2008]
test_data = data[data['Year'] > 2008]

scaler = MinMaxScaler()

train_data_scaled = scaler.fit_transform(train_data['Smoothed_Sunspots'].values.reshape(-1, 1))

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 12

X_train, y_train = create_dataset(train_data_scaled, train_data_scaled, TIME_STEPS)
X_test, y_test = create_dataset(test_data['Smoothed_Sunspots'].values, test_data['Smoothed_Sunspots'].values, TIME_STEPS)

model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping], shuffle=False)

y_pred = model.predict(X_test)

y_train_inv = scaler.inverse_transform(y_train.reshape(1, -1))
y_test_inv = scaler.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = scaler.inverse_transform(y_pred)

# plotujemo stvarne podatke i predikcije

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Smoothed_Sunspots'], label='Smoothed Sunspots', linewidth=2)
plt.plot(test_data['Date'].values[TIME_STEPS:], y_pred_inv, label='Predicted Sunspots', linewidth=2)
plt.title("13-Month Smoothed Sunspot Numbers", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Sunspot Numbers")
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


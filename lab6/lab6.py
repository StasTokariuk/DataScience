import matplotlib
import yfinance as yf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping

# Завантажуємо погодинний курс Біткоїна за останні 730 днів (максимум для погодинних даних)
# Це дасть нам близько 17 500 записів реального ринку
data = yf.download('BTC-USD', period='730d', interval='1h')

# Витягуємо лише ціну закриття та видаляємо порожні значення
prices = data['Close'].dropna().values.reshape(-1, 1)

print(f"Успішно завантажено {len(prices)} реальних вимірів!")

# Нормалізація даних
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

def create_dataset(dataset, look_back=100):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 100
X, Y = create_dataset(prices_scaled, look_back)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = Y[:split], Y[split:]

architectures = [
    {'name': '1L/16N', 'layers': 1, 'neurons': 16, 'dropout': 0.0},
    {'name': '2L/32N', 'layers': 2, 'neurons': 32, 'dropout': 0.0},
    {'name': '3L/64N+Drop', 'layers': 3, 'neurons': 64, 'dropout': 0.2},
    {'name': '4L/64N+Drop', 'layers': 4, 'neurons': 64, 'dropout': 0.3}
]

results = []
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

best_mse = float('inf')
best_predictions = None

print("\nПочаток тестування архітектур.")
for arch in architectures:
    model = Sequential()
    model.add(Input(shape=(look_back,)))

    for _ in range(arch['layers']):
        model.add(Dense(arch['neurons'], activation='relu'))
        if arch['dropout'] > 0:
            model.add(Dropout(arch['dropout']))

    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=100, batch_size=32,validation_split=0.2, callbacks=[early_stop], verbose=0)
    train_time = time.time() - start_time

    predictions = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, predictions)
    params = model.count_params()

    current_predictions = model.predict(X_test, verbose=0)
    current_mse = mean_squared_error(y_test, current_predictions)

    # Зберігаємо прогноз, якщо ця модель найкраща за MSE
    if current_mse < best_mse:
        best_mse = current_mse
        best_predictions = current_predictions
        winner_name = arch['name']

    results.append({
        'Name': arch['name'],
        'MSE': mse,
        'Time (s)': train_time,
        'Params': params,
        'Epochs run': len(history.history['loss'])
    })

df_res = pd.DataFrame(results)

# Нормалізація значень від 0 до 1 для чесного порівняння
mse_norm = (df_res['MSE'] - df_res['MSE'].min()) / (df_res['MSE'].max() - df_res['MSE'].min())
time_norm = (df_res['Time (s)'] - df_res['Time (s)'].min()) / (df_res['Time (s)'].max() - df_res['Time (s)'].min())

df_res['Score'] = (0.6 * mse_norm) + (0.4 * time_norm)

print("\nРезультати R&D дослідження:")
print(df_res.to_string(index=False))

best_model = df_res.loc[df_res['Score'].idxmin()]
print(f"\nПереможець: {best_model['Name']}")

#Візуалізація
fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:red'
ax1.set_xlabel('Архітектури')
ax1.set_ylabel('Помилка MSE', color=color)
ax1.plot(df_res['Name'], df_res['MSE'], marker='o', color=color, linewidth=2)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Час навчання (секунди)', color=color)
ax2.bar(df_res['Name'], df_res['Time (s)'], alpha=0.3, color=color)

plt.title('Аналіз архітектур')
fig.tight_layout()
plt.show()

y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
predictions_real = scaler.inverse_transform(best_predictions.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(y_test_real[-200:], label='Реальна ціна (Факт)', color='black', alpha=0.7)
plt.plot(predictions_real[-200:], label=f'Прогноз ({best_model["Name"]})', color='green', linestyle='--')

plt.title(f'Детальний аналіз прогнозу найкращої архітектури: {best_model["Name"]}')
plt.xlabel('Час (останні 200 годин)')
plt.ylabel('Ціна BTC-USD')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
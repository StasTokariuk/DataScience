import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import matplotlib
matplotlib.use('TkAgg')

FILENAME = 'nbu_usd_history.csv'

# Функція парсингу даних з Мінфіну
def parsed_history_year():
    all_data = []
    years_months = [
        "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08",
        "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"
    ]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for ym in years_months:
        url = f"https://index.minfin.com.ua/ua/exchange/nbu/curr/usd/{ym}/"
        print(f"Парсинг даних за {ym}...")
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', class_='zebra')
            if not table: continue
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    date_element = cols[0].find(string=True)
                    if date_element:
                        date_raw = date_element.strip()
                        price_raw = cols[1].get_text(strip=True).replace(',', '.')
                        try:
                            all_data.append({
                                'Date': datetime.strptime(date_raw, '%d.%m.%Y'),
                                'Price': float(price_raw)
                            })
                        except ValueError:
                            continue
            time.sleep(0.3)
        except Exception as e:
            print(f"Помилка при запиті {ym}: {e}")

    df = pd.DataFrame(all_data).sort_values('Date').reset_index(drop=True)
    df.to_csv(FILENAME, index=False)
    return df


# Автоматична генерація аномалій
def inject_anomalies(series, rate=0.04, severity=4.0):
    data = series.copy()
    n_anomalies = int(len(data) * rate)
    std_dev = np.std(data)
    indices = np.random.choice(range(5, len(data) - 5), n_anomalies, replace=False)

    for idx in indices:
        direction = np.random.choice([1, -1])
        data[idx] += direction * (severity * std_dev + np.random.uniform(0.5, 1.5))
    return data, indices


# Алгоритм детекції аномалій
def anomaly_detector(series, window=10, k=3.0):
    rolling_med = series.rolling(window=window, center=True).median().fillna(method='bfill').fillna(method='ffill')

    def get_mad(x):
        median = np.median(x)
        return np.median(np.abs(x - median))

    rolling_mad = series.rolling(window=window, center=True).apply(get_mad).fillna(method='bfill').fillna(method='ffill')

    anomalies = np.where(np.abs(series - rolling_med) > (k * rolling_mad * 1.4826))[0]

    cleaned = series.copy()
    for idx in anomalies:
        cleaned[idx] = rolling_med[idx]

    return anomalies, cleaned

# Рекурентне згладжування alfa-beta-gamma фільтром
def abg_filter(data, alpha=0.4, beta=0.05, gamma=0.01):
    x, v, a = data[0], 0, 0
    dt = 1
    results = []
    for z in data:
        x_p = x + v * dt + 0.5 * a * dt ** 2
        v_p = v + a * dt
        rk = z - x_p
        x = x_p + alpha * rk
        v = v_p + (beta / dt) * rk
        a = a + (gamma / (2 * dt ** 2)) * rk
        results.append(x)
    return np.array(results)


if os.path.exists(FILENAME):
    df = pd.read_csv(FILENAME)
    df['Date'] = pd.to_datetime(df['Date'])
else:
    print(f"Файл {FILENAME} не знайдено. Запуск парсингу.")
    df = parsed_history_year()


df['Price_Anom'], true_idx = inject_anomalies(df['Price'].values)
det_idx, df['Price_Clean'] = anomaly_detector(df['Price_Anom'])
df['Price_Filtered'] = abg_filter(df['Price_Clean'].values)

# Візуалізація
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price_Anom'], 'r.', alpha=0.3, label='Дані з аномаліями')
plt.plot(df['Date'], df['Price_Filtered'], 'b-', label='abg Фільтр')
plt.scatter(df.iloc[det_idx]['Date'], df.iloc[det_idx]['Price_Anom'], color='black', marker='x')
plt.legend()
plt.show()

mse = np.mean((df['Price'] - df['Price_Filtered']) ** 2)
print(f"MSE: {mse:.6f}")
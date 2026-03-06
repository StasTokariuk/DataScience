import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import matplotlib
matplotlib.use('TkAgg')

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
            if not table:
                continue

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

            time.sleep(0.5)
        except Exception as e:
            print(f"Помилка при запиті {ym}: {e}")

    if not all_data:
        print("\nДані не зібрано!")
        return pd.DataFrame(columns=['Date', 'Price'])

    df = pd.DataFrame(all_data)
    df = df.sort_values('Date').reset_index(drop=True)
    df.to_csv('nbu_usd_history.csv', index=False)
    return df


# Отримуємо дані
df = parsed_history_year()
prices = df['Price'].values
dates = df['Date']
x = np.arange(len(prices))

# Статистичні характеристики
mean_val = np.mean(prices)
variance_val = np.var(prices)
std_dev = np.std(prices)

print(f"\nСтатистика за рік:")
print(f"Математичне очікування: {mean_val:.4f}")
print(f"Дисперсія: {variance_val:.4f}")
print(f"Стандартне відхилення: {std_dev:.4f}")

# Кубічний тренд та синтезована математична модель
# Кубічна апроксимація
poly_coeffs = np.polyfit(x, prices, 3)
trend_line = np.polyval(poly_coeffs, x)


residuals = prices - trend_line
noise_std = np.std(residuals)
synthetic_noise = np.random.normal(0, noise_std, len(prices))
synthetic_model = trend_line + synthetic_noise

# Візуалізація
plt.figure(figsize=(14, 7))

# Головний графік
plt.plot(dates, prices, label='Реальний курс (Парсинг)', color='royalblue', alpha=0.5, linewidth=1.5)
plt.plot(dates, trend_line, label='Кубічний тренд (Апроксимація)', color='crimson', linewidth=3)
plt.plot(dates, synthetic_model, '--', label='Синтезована модель', color='seagreen', alpha=0.7)

plt.title('Аналіз курсу USD/UAH за рік: Парсинг, Тренди та Моделювання', fontsize=14)
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Курс (UAH)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Реальна похибка = Реальна ціна - Лінія тренду
real_residuals = prices - trend_line
real_std = np.std(real_residuals)
real_mean_error = np.mean(real_residuals)

synthetic_noise = np.random.normal(0, real_std, len(prices))
synthetic_data = trend_line + synthetic_noise

# Порівняння характеристик
print("\nПорівняння характеристик")
print(f"{'Параметр':<25} | {'Реальні дані':<15} | {'Синтезована модель':<15}")
print("-" * 65)
print(f"{'Мат. очікування (ціна)':<25} | {np.mean(prices):<15.4f} | {np.mean(synthetic_data):<15.4f}")
print(f"{'Стандартне відхилення':<25} | {real_std:<15.4f} | {np.std(synthetic_noise):<15.4f}")

# Візуалізація
plt.figure(figsize=(15, 6))

# Лівий графік
plt.subplot(1, 2, 1)
plt.plot(dates, prices, label='Реальний курс (Parsed)', color='blue', alpha=0.5)
plt.plot(dates, synthetic_data, label='Синтетична модель', color='green', linestyle='--', alpha=0.7)
plt.plot(dates, trend_line, label='Кубічний тренд', color='red', linewidth=2)
plt.title('Порівняння реальних та синтезованих даних')
plt.legend()
plt.xticks(rotation=45)

# Правий графік
plt.subplot(1, 2, 2)
plt.hist(real_residuals, bins=20, alpha=0.5, label='Реальна похибка', density=True, color='blue')
plt.hist(synthetic_noise, bins=20, alpha=0.5, label='Синтетичний шум', density=True, color='green')
plt.title('Визначення закону розподілу (Нормальний)')
plt.legend()

plt.tight_layout()
plt.show()
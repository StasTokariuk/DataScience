import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
import matplotlib
matplotlib.use('TkAgg')

# Зчитування вхідних даних з Excel
try:
    df_ssd = pd.read_excel('ssd_data.xlsx')
except FileNotFoundError:
    print("Файл не знайдено")

# MCDA
weights = np.array([0.2, 0.15, 0.3, 0.2, 0.15]) # Ваги критеріїв
norm_df = df_ssd.copy()

# Нормалізація: максимізація швидкості та ресурсу, мінімізація ціни
for col in ['Read_Speed', 'Write_Speed', 'TBW', 'IOPS']:
    norm_df[col] = df_ssd[col] / df_ssd[col].max()
norm_df['Price'] = df_ssd['Price'].min() / df_ssd['Price']

df_ssd['Score'] = norm_df[['Read_Speed', 'Write_Speed', 'Price', 'TBW', 'IOPS']].dot(weights)
df_results = df_ssd.sort_values(by='Score', ascending=False)

# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.bar(df_results['Model'], df_results['Score'], color='skyblue')
plt.title('Інтегральна оцінка ефективності SSD (MCDA)')
plt.ylabel('Оцінка')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Розв'язання LP задачі
model = cp_model.CpModel()
x1 = model.NewIntVar(0, 10, 'x1')
x2 = model.NewIntVar(0, 10, 'x2')

model.Add(3 * x1 + 4 * x2 <= 24)
model.Add(x1 + 2 * x2 <= 8)
model.Add(x1 <= 4)
model.Add(x2 <= 3)

model.Maximize(2 * x1 + 2 * x2)

solver = cp_model.CpSolver()
if solver.Solve(model) == cp_model.OPTIMAL:
    lp_res = (solver.Value(x1), solver.Value(x2), -2*(solver.Value(x1) + solver.Value(x2)))
    res_x1 = solver.Value(x1)
    res_x2 = solver.Value(x2)
    q_val = -2 * (res_x1 + res_x2)

    print(f"Оптимальна кількість X1: {res_x1}")
    print(f"Оптимальна кількість X2: {res_x2}")
    print(f"Мінімальне значення Q:  {q_val}")

# Графічна верифікація LP
x_vals = np.linspace(0, 8, 400)
plt.figure()
plt.plot(x_vals, (12 - 1.5*x_vals)/2, label='1.5*X1 + 2*X2 <= 12')
plt.plot(x_vals, (8 - x_vals)/2, label='X1 + 2*X2 <= 8')
plt.axhline(y=3, color='r', linestyle='--', label='X2 <= 3')
plt.axvline(x=4, color='g', linestyle='--', label='X1 <= 4')
plt.scatter(lp_res[0], lp_res[1], color='red', s=100, zorder=5, label=f'Оптимум ({lp_res[0]}, {lp_res[1]})')
plt.title('Графічний метод розв’язку LP')
plt.legend()
plt.grid(True)
plt.show()
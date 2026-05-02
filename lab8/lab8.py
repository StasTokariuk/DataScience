import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
matplotlib.use('TkAgg')

# Підготовка вхідних даних
print(" Парсинг та підготовка даних")
# Парсинг файлів
df = pd.read_excel('sample_data.xlsx')
descriptions = pd.read_excel('data_description.xlsx')

# Сегментація ознак
valid_places = ['Вказує позичальник', 'параметри, повязані з виданим продуктом']
client_bank_descriptions = descriptions[descriptions['Place_of_definition'].isin(valid_places)]
client_bank_fields = client_bank_descriptions["Field_in_data"].tolist()

# Знаходимо перетин індикаторів та залишаємо колонки без суцільних пропусків
col_intersection = list(set(client_bank_fields).intersection(df.columns))
data = df.loc[:, col_intersection].dropna(axis=1)

# Формування цільової бінарної змінної
if 'loan_overdue' not in data.columns and 'loan_overdue' in df.columns:
    data['loan_overdue'] = df['loan_overdue']

data['give'] = 1 - data['loan_overdue']

# Відбір фічів
X = data.drop(columns=['loan_overdue', 'give'], errors='ignore')

# Конвертація в числа та заповнення пропусків медіаною
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X.fillna(X.median(), inplace=True)

# Нормалізація
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Розмірність підготовлених даних: {X_scaled.shape}")

# Формування скорингової моделі (SVM)
print("\n Скоринговий аналіз (SVC)")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, data['give'], test_size=0.2, random_state=42)

svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print(f"Точність моделі опорних векторів: {accuracy_score(y_test, y_pred):.4f}")
print("Звіт класифікації:\n", classification_report(y_test, y_pred))


# Виявлення шахрайства
print("\nВиявлення фальсифікацій (One-Class SVM)")
outliers_fraction = 0.1 # Згідно з лекцією (10% аномалій)
oc_svm = OneClassSVM(kernel='rbf', nu=outliers_fraction)
anomaly_preds = oc_svm.fit_predict(X_scaled)

data['is_fraud_suspected'] = np.where(anomaly_preds == -1, 1, 0)
data['predicted_return'] = svm_model.predict(X_scaled)

num_frauds = data['is_fraud_suspected'].sum()
print(f"Виявлено {num_frauds} підозрілих заявок з {len(data)} ({(num_frauds/len(data)):.1%}).")

# Візуалізація результатів
plt.figure(figsize=(12, 5))

# Прогноз
plt.subplot(1, 2, 1)
data['predicted_return'].value_counts().plot(kind='bar', color=['#4CAF50', '#F44336'])
plt.title('Прогноз повернення (SVC)')
plt.xticks([0, 1], ['Поверне (1)', 'Не поверне (0)'], rotation=0)

# Шахрайство
plt.subplot(1, 2, 2)
data['is_fraud_suspected'].value_counts().plot(kind='bar', color=['#2196F3', '#FF9800'])
plt.title('Виявлення шахрайства (One-Class SVM)')
plt.xticks([0, 1], ['Норма (0)', 'Шахрайство (1)'], rotation=0)

plt.tight_layout()
plt.show()

data.to_excel('scoring_results.xlsx', index=False)
print("\nРезультати збережено у 'scoring_results_svm.xlsx'")
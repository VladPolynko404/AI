import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import os

# --- АВТОМАТИЧНЕ ВИЗНАЧЕННЯ ШЛЯХУ ---
# Знаходимо папку, в якій лежить цей скрипт
current_dir = os.path.dirname(os.path.abspath(__file__))
# Формуємо повний шлях до файлу Variant 1
input_file = os.path.join(current_dir, 'data_regr_1.txt')

# Перевірка наявності файлу
if not os.path.exists(input_file):
    print(f"ПОМИЛКА: Файл не знайдено за шляхом: {input_file}")
    exit()

# Завантаження даних (Варіант 1)
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка на навчальний та тестовий набори
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Створення та навчання моделі
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Прогнозування
y_test_pred = regressor.predict(X_test)

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_test_pred, color='red', linewidth=3, label='Regression line')
plt.title('Regression for Variant 1')
plt.xlabel('Input')
plt.ylabel('Target')
plt.legend()
plt.show()

# Вивід метрик
print("Performance for Variant 1:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
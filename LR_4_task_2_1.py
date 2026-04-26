import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import os

# --- ИСПРАВЛЕННЫЙ БЛОК ПУТИ ---
# Определяем директорию, где лежит сам файл 2_1.py
current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_dir, 'data_singlevar_regr.txt')

# Проверка: существует ли файл?
if not os.path.exists(input_file):
    print(f"ОШИБКА: Файл '{input_file}' не найден!")
    print(f"Убедитесь, что текстовый файл лежит в папке: {current_dir}")
    exit() # Останавливаем выполнение, если файла нет
# ------------------------------

# Завантаження даних
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
plt.scatter(X_test, y_test, color='green', label='Actual data')
plt.plot(X_test, y_test_pred, color='black', linewidth=3, label='Regression line')
plt.title('Linear Regression: Single Variable')
plt.xlabel('Input variable')
plt.ylabel('Target variable')
plt.legend()
plt.show()

# Метрики
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Збереження моделі
model_path = os.path.join(current_dir, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(regressor, f)
    print(f"\nМодель сохранена как: {model_path}")
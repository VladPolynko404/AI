import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve

# 1. Генерація даних для Варіанту 1
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# 2. Функція для побудови кривих навчання
def plot_learning_curves(model, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='neg_mean_squared_error')
    
    train_errors = np.sqrt(-train_scores.mean(axis=1))
    val_errors = np.sqrt(-val_scores.mean(axis=1))
    
    plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="Навчання")
    plt.plot(train_sizes, val_errors, "b-", linewidth=3, label="Валідація")
    plt.title(title)
    plt.xlabel("Розмір навчального набору")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()

# --- Побудова кривих для ЛІНІЙНОЇ моделі (буде недонавчання) ---
print("Будуємо криві для лінійної моделі...")
plot_learning_curves(LinearRegression(), X, y, "Learning Curves (Linear - Underfitting)")

# --- Побудова кривих для ПОЛІНОМІАЛЬНОЇ моделі (2 ступінь - ідеально) ---
print("Будуємо криві для полінома 2-го ступеня...")
poly_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_2.fit_transform(X)
plot_learning_curves(LinearRegression(), X_poly, y, "Learning Curves (Polynomial deg 2 - Optimal)")
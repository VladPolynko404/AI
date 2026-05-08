import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# 1. Завантаження вхідних даних
# Цей блок автоматично знайде файл, якщо він лежить в тій же папці, що і цей скрипт
script_dir = os.path.dirname(__file__)  # Шлях до папки зі скриптом
file_path = os.path.join(script_dir, 'data_clustering.txt')

try:
    X = np.loadtxt(file_path, delimiter=',')
except FileNotFoundError:
    print(f"Помилка: Файл не знайдено за шляхом {file_path}")
    print("Переконайся, що файл data_clustering.txt лежить у тій же папці, що і цей скрипт.")
    exit()

num_clusters = 5 

# 2. Створення та навчання моделі KMeans
# n_init='auto' або 10 — залежить від версії sklearn, зазвичай зараз краще 'auto'
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10, random_state=42)
kmeans.fit(X)

# 3. Візуалізація меж кластерів
step_size = 0.01

# Визначаємо межі графіка
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Створюємо сітку
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), 
                             np.arange(y_min, y_max, step_size))

# Передбачення міток для всіх точок сітки
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)

# 4. Побудова графіка
plt.figure(figsize=(10, 7))

# Відображення фонових областей кластерів
plt.imshow(output, interpolation='nearest', 
           extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()), 
           cmap=plt.cm.Paired, aspect='auto', origin='lower')

# Відображення вхідних точок (даних)
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=30, alpha=0.5)

# Відображення центрів кластерів
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=100, 
            linewidths=3, color='black', zorder=12)

plt.title('Межі кластерів та центроїди (K-Means)')
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.show()
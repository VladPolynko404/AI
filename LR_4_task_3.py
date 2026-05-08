import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
import os

# 1. Автоматичне визначення шляху до файлу
# Отримуємо шлях до папки, де лежить цей скрипт
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'data_clustering.txt')

# Завантаження даних з перевіркою
try:
    X = np.loadtxt(file_path, delimiter=',')
except FileNotFoundError:
    print(f"Помилка: Файл не знайдено за шляхом: {file_path}")
    print("Переконайся, що 'data_clustering.txt' лежить в тій же папці, що і цей .py файл.")
    exit()

# 2. Оцінка ширини вікна (bandwidth)
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# 3. Навчання моделі MeanShift
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# 4. Витягування центрів та кількості кластерів
cluster_centers = meanshift_model.cluster_centers_
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))

print(f"Кількість знайдених кластерів: {num_clusters}")
print(f"Центри кластерів:\n{cluster_centers}")

# 5. Візуалізація
plt.figure(figsize=(10, 7))
# Додаємо більше маркерів на випадок, якщо кластерів буде багато
markers = 'o*xvsD^P' 
colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))

for i, marker, color in zip(range(num_clusters), markers, colors):
    # Малюємо точки кластера
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color=color, label=f'Cluster {i}')
    
    # Малювання центру кожного кластера
    center = cluster_centers[i]
    plt.plot(center[0], center[1], marker='o', markerfacecolor='red', 
             markeredgecolor='black', markersize=12, markeredgewidth=2)

plt.title(f'Кластеризація Mean Shift (Знайдено кластерів: {num_clusters})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
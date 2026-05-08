import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Завантаження вхідних даних [cite: 178, 179]
X = np.loadtxt('data_clustering.txt', delimiter=',')
num_clusters = 5  # Кількість кластерів [cite: 181]

# 2. Створення та навчання моделі KMeans [cite: 198, 200]
# Використовуємо k-means++ для кращої ініціалізації центроїдів 
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
kmeans.fit(X)

# 3. Візуалізація меж кластерів [cite: 201]
step_size = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), 
                             np.arange(y_min, y_max, step_size))

# Передбачення міток для всієї сітки [cite: 215]
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)

# Побудова графіка [cite: 220, 228, 232]
plt.figure()
plt.imshow(output, interpolation='nearest', extent=(x_vals.min(), x_vals.max(), 
           y_vals.min(), y_vals.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')

# Відображення вхідних точок
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)

# Відображення центрів кластерів [cite: 234, 237]
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', s=210, 
            linewidths=4, color='black', zorder=12, facecolors='black')

plt.title('Межі кластерів (K-Means)')
plt.show()
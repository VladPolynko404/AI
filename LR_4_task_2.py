import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Завантаження вбудованого набору даних Iris [cite: 265]
iris = load_iris()
X = iris.data  # Атрибути: довжина/ширина чашолистка та пелюстки [cite: 257]

# Створення моделі для 3-х кластерів (оскільки відомо 3 типи ірисів) [cite: 257, 274]
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
# Навчання моделі [cite: 276]
kmeans.fit(X)
# Прогнозування приналежності до кластерів [cite: 278]
y_kmeans = kmeans.predict(X)

# Візуалізація результатів (перші дві ознаки) [cite: 280]
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Відображення знайдених центроїдів [cite: 281, 282]
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, label='Centroids')

plt.title('Кластеризація Iris (K-Means)')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()
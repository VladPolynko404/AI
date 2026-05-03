import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y, title=''):
    # Визначення меж для візуалізації
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    
    # Крок сітки
    mesh_step_size = 0.01
    x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), 
                                 np.arange(y_min, y_max, mesh_step_size))
    
    # Прогноз для всієї сітки
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    
    # Малювання
    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.Paired, shading='auto')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.title(title)
    plt.show()

if __name__ == "__main__":    
    print("Функція візуалізації готова до використання.")
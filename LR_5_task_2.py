import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities import visualize_classifier

# Завантаження даних
try:
    data = np.loadtxt('data_imbalance.txt', delimiter=',')
except:
    print("Файл data_imbalance.txt не знайдено!")
    exit()

X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

if len(sys.argv) > 1 and sys.argv[1] == 'balance':
    params['class_weight'] = 'balanced'

classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)

visualize_classifier(classifier, X_test, y_test, 'Результат класифікації')

y_test_pred = classifier.predict(X_test)
print("\n" + "#"*40)
print("Performance report:\n")
print(classification_report(y_test, y_test_pred, zero_division=0))
print("#"*40)
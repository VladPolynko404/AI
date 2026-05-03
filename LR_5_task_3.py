import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report

data = np.loadtxt('data_random_forests.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Сітка параметрів
parameter_grid = [
    {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
    {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
]

metrics = ['precision_weighted', 'recall_weighted']

for metric in metrics:
    print(f"\n##### Пошук оптимальних параметрів для {metric}")
    classifier = GridSearchCV(ExtraTreesClassifier(random_state=0), 
                              parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

    print("\nСітка результатів:")
    for params, avg_score in zip(classifier.cv_results_['params'], classifier.cv_results_['mean_test_score']):
        print(f"{params} --> {round(avg_score, 3)}")
    
    print(f"\nНайкращі параметри: {classifier.best_params_}")
    
    y_pred = classifier.predict(X_test)
    print("\nЗвіт по класифікації:\n")
    print(classification_report(y_test, y_pred))
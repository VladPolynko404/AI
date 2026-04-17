# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def main():
    # 1. Завантаження даних (Iris dataset)
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 2. Розбиття на тренувальну і тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5
    )

    # 3. Створення моделі (RBF kernel)
    classifier = SVC(kernel='rbf')

    # 4. Навчання
    classifier.fit(X_train, y_train)

    # 5. Прогноз
    y_pred = classifier.predict(X_test)

    # 6. Результати
    print("Iris Classification Results:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
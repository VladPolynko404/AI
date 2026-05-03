import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from utilities import visualize_classifier

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Classify data using Ensemble Learning')
    parser.add_argument('--classifier-type', dest='classifier_type', required=True,
                        choices=['rf', 'erf'], help="Type of classifier: 'rf' or 'erf'")
    return parser

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type


    try:
        data = np.loadtxt('data_random_forests.txt', delimiter=',')
    except:
        print("Файл data_random_forests.txt не знайдено!")
        exit()

    X, y = data[:, :-1], data[:, -1]

    # Розбивка на класи для візуалізації вхідних даних
    class_0 = X[y==0]
    class_1 = X[y==1]
    class_2 = X[y==2]

    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white', edgecolors='black', marker='s')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', marker='o')
    plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white', edgecolors='black', marker='^')
    plt.title('Вхідні дані')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(**params)
    else:
        classifier = ExtraTreesClassifier(**params)

    classifier.fit(X_train, y_train)
    visualize_classifier(classifier, X_train, y_train, 'Навчальний набір')

    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test, 'Тестовий набір')

    class_names = ['Class-0', 'Class-1', 'Class-2']
    print("\n" + "#"*40)
    print("Результати на тестовому наборі:\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#"*40)

    # Оцінка рівнів довіри
    test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])
    print("\nConfidence measure (Рівні довіри):")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = 'Class-' + str(np.argmax(probabilities))
        print(f'Точка: {datapoint} -> Клас: {predicted_class}, Ймовірності: {probabilities}')
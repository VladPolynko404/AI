import numpy as np
import os

from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler

#  Путь к файлу
base_dir = os.path.dirname(__file__)
input_file = os.path.join(base_dir, "income_data.txt")

X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

#  Чтение данных
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line.strip().split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

#  Кодирование
label_encoders = []
X_encoded = np.empty(X.shape, dtype=int)

for i in range(X.shape[1]):
    try:
        X_encoded[:, i] = X[:, i].astype(int)
    except:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoders.append(le)

# Разделение признаков и меток
X = X_encoded[:, :-1]
y = X_encoded[:, -1]

#  НОРМАЛИЗАЦИЯ (важно!)
scaler = StandardScaler()
X = scaler.fit_transform(X)

#  Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

#  Улучшенный классификатор
classifier = OneVsOneClassifier(
    LinearSVC(random_state=0, max_iter=10000)
)

classifier.fit(X_train, y_train)

#  Предсказание
y_pred = classifier.predict(X_test)

# Метрики
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1.mean())

#  Тестовая точка
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

input_encoded = []
encoder_index = 0


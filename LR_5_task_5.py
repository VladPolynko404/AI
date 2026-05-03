import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

input_file = 'traffic_data.txt'
data = []

# Якщо файлу немає, створіть його або завантажте. Приклад структури: Tuesday,00:00,Atlanta,no,15
try:
    with open(input_file, 'r') as f:
        for line in f.readlines():
            items = line.strip().split(',')
            data.append(items)
    data = np.array(data)
except:
    print("Файл traffic_data.txt не знайдено!")
    exit()

# Кодування нечислових ознак
label_encoders = []
X_encoded = np.empty(data.shape)

for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(data[:, i])
        label_encoders.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

regressor = ExtraTreesRegressor(n_estimators=100, max_depth=4, random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(f"Mean Absolute Error: {round(mean_absolute_error(y_test, y_pred), 2)}")

# Тест на одиночній точці
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint_encoded = []
count = 0
for i, item in enumerate(test_datapoint):
    if item.isdigit():
        test_datapoint_encoded.append(int(item))
    else:
        val = label_encoders[count].transform([item])[0]
        test_datapoint_encoded.append(val)
        count += 1

predicted_traffic = regressor.predict([test_datapoint_encoded])[0]
print(f"Predicted traffic: {int(predicted_traffic)}")
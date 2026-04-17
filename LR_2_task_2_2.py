from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 1. Завантажуємо дані
data = load_iris()
X = data.data
y = data.target

# 2. Ділимо на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Створюємо модель
classifier = SVC(kernel='rbf')

# 4. Навчаємо
classifier.fit(X_train, y_train)

# 5. Прогнозуємо
y_pred = classifier.predict(X_test)

# 6. Оцінка
print("RBF Kernel Results:")
print(classification_report(y_test, y_pred))
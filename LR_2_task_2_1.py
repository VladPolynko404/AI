from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Завантажуємо готовий датасет (іриси)
data = datasets.load_iris()
X = data.data
y = data.target

# Розбиваємо на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Створюємо модель (поліноміальне ядро)
classifier = SVC(kernel='poly', degree=8)

# Навчання
classifier.fit(X_train, y_train)

# Передбачення
y_pred = classifier.predict(X_test)

# Результат
print("Polynomial Kernel Results:")
print(classification_report(y_test, y_pred))
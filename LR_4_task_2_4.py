from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0), color='skyblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
plt.xlabel('Виміряно (Measured)')
plt.ylabel('Передбачено (Predicted)')
plt.title('Diabetes Dataset: Measured vs Predicted')
plt.show()
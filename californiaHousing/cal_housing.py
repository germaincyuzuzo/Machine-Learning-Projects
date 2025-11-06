from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
housing = fetch_california_housing(as_frame=True)
data = housing.frame

X = data.drop(columns=['MedHouseVal'])
y = data['MedHouseVal']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

#Train model
model = LinearRegression()
model.fit(X_train, y_train)

print(f'R² Score: {model.score(X_test, y_test)}')

y_pred = model.predict(X_test)
print(y_pred)

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Loading data
d1=pd.read_csv("student-mat.csv",sep=";")
d2=pd.read_csv("student-por.csv",sep=";")

data=pd.concat([d1, d2])

X = data[['G1','G2']]
y = data['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Model Instantiation
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
y_pred = model.predict(X_test)

# Visualizing predictions
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual G3 Grades')
plt.ylabel('Predicted G3 Grades')
plt.title('Actual vs Predicted Grades')
plt.show()

#Calculating MSE and MAE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE: {mse:.3f}')
print(f'MAE: {mae:.3f}')
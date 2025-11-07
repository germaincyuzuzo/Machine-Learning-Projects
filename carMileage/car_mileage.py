import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


#Loading data
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
data = pd.read_csv('auto-mpg.data', delim_whitespace=True, names=column_names, na_values='?')

data = data.dropna()

# print(data.isna().sum())
# print(data.describe())


X = data.drop(columns=['mpg', 'car_name'])
y = data['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

#Train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)


#Train Ridge Regression
ridge_reg = Ridge(alpha=5)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)


#Train Lasso Regression
lasso_reg = Lasso(alpha=7)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_test)


#Comparison between models
models = {
    "Linear": (y_pred_lin, lin_reg),
    "Ridge": (y_pred_ridge, ridge_reg),
    "Lasso": (y_pred_lasso, lasso_reg)
}

for name, (pred, model) in models.items():
    r2 = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"{name} Regression --> R2: {r2}, MSE: {mse}, MAE: {mae}")


#Data frame comparing coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Linear': lin_reg.coef_,
    'Ridge': ridge_reg.coef_,
    'Lasso': lasso_reg.coef_
})

print(coefficients)

#Visualize the coefficients
coefficients.set_index('Feature').plot(kind='bar', figsize=(10,6))
plt.title('Feature Coefficients Comparison')
plt.ylabel('Coefficient Value')
plt.xlabel('Feature')
plt.xticks(rotation=45)
plt.show()

# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.xlabel('Actual mileage')
# plt.ylabel('Predicted mileage')
# plt.title('Actual vs Predicted Mileage')
# plt.show()
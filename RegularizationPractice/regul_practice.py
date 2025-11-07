from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
import matplotlib.pyplot as plt


#Loading the dataset
housing = fetch_california_housing(as_frame=True)
data = housing.frame

X = data.drop(columns=['MedHouseVal'])
y = data['MedHouseVal']

#Cleaning the dataset
print(data.head())
print(data.isnull().sum())
print(data.isna().sum())

#Splitting and fitting the models
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

#Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_reg_pred = lin_reg.predict(X_test)

#Ridge model
ridge_reg = Ridge(alpha=0.1)
ridge_reg.fit(X_train, y_train)
y_rid_pred = ridge_reg.predict(X_test)

#Lasso model
lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(X_train, y_train)
y_lasso_pred = lasso_reg.predict(X_test)

models = {
    "Linear": (y_reg_pred, lin_reg),
    "Ridge": (y_rid_pred, ridge_reg),
    "Lasso": (y_lasso_pred, lasso_reg)
}

for name, (pred, model) in models.items():
    r2 = model.score(X_test, y_test)
    # mse = mean_squared_error(y_test, pred)
    # mae = mean_absolute_error(y_test, pred)
    print(f"{name} Regression --> R2: {r2}")

#Visualizing the training against test R2 



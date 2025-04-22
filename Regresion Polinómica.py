# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('calories.csv')
X = dataset[['Session_Duration (hours)']].values
y = dataset['Calories_Burned'].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizar resultados de la regresión lineal
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Calorías Quemadas vs Duración (Lineal)')
plt.xlabel('Duración de sesión (horas)')
plt.ylabel('Calorías quemadas')
plt.show()

# Visualizar resultados de la regresión polinómica
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Calorías Quemadas vs Duración (Polinómica)')
plt.xlabel('Duración de sesión (horas)')
plt.ylabel('Calorías quemadas')
plt.show()

# Visualización más suave
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.transform(X_grid)), color='blue')
plt.title('Calorías Quemadas vs Duración (Curva suave)')
plt.xlabel('Duración de sesión (horas)')
plt.ylabel('Calorías quemadas')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
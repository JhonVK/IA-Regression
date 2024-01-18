import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

base_saude2= pd.read_csv(r'C:\Users\joaov\OneDrive\Área de Trabalho\I.A-Classificação\CSV\plano_saude2.csv')

x=base_saude2.iloc[:, 0:1].values
y=base_saude2.iloc[:, 1].values

print(base_saude2)


poly= PolynomialFeatures(degree=4)

x_poly= poly.fit_transform(x)

print(x_poly)

regressor_polinomial= LinearRegression()
regressor_polinomial.fit(x_poly, y)

print(regressor_polinomial.intercept_) #b0
print(regressor_polinomial.coef_) #b1

novo_reg=[[40]]
novo_reg=poly.transform(novo_reg)

print(regressor_polinomial.predict(novo_reg))

previsoes= regressor_polinomial.predict(x_poly)
print(previsoes)

grafico = px.scatter(x = x[:,0], y = y)
grafico.add_scatter(x = x[:,0], y = previsoes, name = 'Regressão')
grafico.show()
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

base_saude2= pd.read_csv(r'C:\Users\joaov\OneDrive\Área de Trabalho\I.A-Classificação\CSV\plano_saude2.csv')

x_saude=base_saude2.iloc[:, 0:1].values
y_saude=base_saude2.iloc[:, 1].values

from sklearn.tree import DecisionTreeRegressor

regressor_arvore_saude= DecisionTreeRegressor()
regressor_arvore_saude.fit(x_saude, y_saude)

previsoes=regressor_arvore_saude.predict(x_saude)
print(previsoes)
print(y_saude)

score= regressor_arvore_saude.score(x_saude, y_saude)
print(score)
##precisao de 100%

grafico = px.scatter(x = x_saude.ravel(), y = y_saude)
grafico.add_scatter(x = x_saude.ravel(), y = previsoes, name = 'Regressão')
grafico.show()


x_teste_arvore= np.arange(min(x_saude), max(x_saude), 0.1)
print(x_teste_arvore)
x_teste_arvore=x_teste_arvore.reshape(-1,1)

grafico = px.scatter(x = x_saude.ravel(), y = y_saude)
grafico.add_scatter(x = x_teste_arvore.ravel(), y = regressor_arvore_saude.predict(x_teste_arvore), name = 'Regressão')
grafico.show()
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.svm import SVR


base_house = pd.read_csv(
    r'C:\Users\joaov\OneDrive\Área de Trabalho\I.A-Classificação\CSV\house_prices.csv')
base_house.drop('date', axis=1, inplace=True)

x=base_house.iloc[:, 2:18].values
print(x)

y=base_house.iloc[:, 1].values



from sklearn.model_selection import train_test_split
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)

print(X_casas_treinamento.shape)
from sklearn.preprocessing import StandardScaler

x_escalonado=StandardScaler()
x_treinamento_escalonado=x_escalonado.fit_transform(X_casas_treinamento)
y_escalonado=StandardScaler()
y_treinamento_escalonado=y_escalonado.fit_transform(y_casas_treinamento.reshape(-1,1))


x_teste_escalonado=x_escalonado.transform(X_casas_teste)
y_teste_escalonado=y_escalonado.transform(y_casas_teste.reshape(-1,1))

regressor_svr_casas= SVR(kernel='rbf')
regressor_svr_casas.fit(x_treinamento_escalonado, y_treinamento_escalonado.ravel())

print(regressor_svr_casas.score(x_teste_escalonado, y_teste_escalonado))

print(regressor_svr_casas.score(x_treinamento_escalonado, y_treinamento_escalonado))

previsoes = regressor_svr_casas.predict(x_teste_escalonado).reshape(-1, 1)
print(previsoes)
#desescalonar

y_casas_inverse=y_escalonado.inverse_transform(y_teste_escalonado)
previsoes_inverse=y_escalonado.inverse_transform(previsoes)

print(y_casas_inverse)
print(previsoes_inverse)

from sklearn.metrics import mean_absolute_error
abso=mean_absolute_error(y_casas_inverse, previsoes_inverse)

print(abso)
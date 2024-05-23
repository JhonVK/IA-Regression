import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

base_house= pd.read_csv(r'C:\Users\joaov\Desktop\OneDrive\CSV\house_prices.csv')


x=base_house.iloc[:, 3:19].values

y=base_house.iloc[:, 2].values


from sklearn.model_selection import train_test_split
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)


print(X_casas_treinamento.shape)
print(X_casas_teste.shape)

from sklearn.preprocessing import StandardScaler

x_escalonado=StandardScaler()
x_treinamento_escalonado=x_escalonado.fit_transform(X_casas_treinamento)
y_escalonado=StandardScaler()
y_treinamento_escalonado=y_escalonado.fit_transform(y_casas_treinamento.reshape(-1,1))


x_teste_escalonado=x_escalonado.transform(X_casas_teste)
y_teste_escalonado=y_escalonado.transform(y_casas_teste.reshape(-1,1))


#(16+1)/2
regressor_rna_casas=MLPRegressor(max_iter=1000, hidden_layer_sizes=(9, 9))
regressor_rna_casas.fit(x_treinamento_escalonado, y_treinamento_escalonado.ravel())

score=regressor_rna_casas.score(x_teste_escalonado, y_teste_escalonado)
print(score)

previsoes=regressor_rna_casas.predict(x_teste_escalonado)
print(previsoes)

previsoes_des=y_escalonado.inverse_transform(previsoes.reshape(-1,1))
print(previsoes_des)

print(y_casas_teste)

from sklearn.metrics import mean_absolute_error
abso=mean_absolute_error(y_casas_teste, previsoes_des)

print(abso)
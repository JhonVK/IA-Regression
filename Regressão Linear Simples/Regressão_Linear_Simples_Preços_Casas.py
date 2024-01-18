import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



base_house= pd.read_csv(r'C:\Users\joaov\OneDrive\Área de Trabalho\I.A-Classificação\CSV\house_prices.csv')
print(base_house)

print(base_house.describe())

print(base_house.isnull().sum())

base_house.drop('date', axis=1, inplace=True)

print(base_house.corr())
figura=plt.figure(figsize=(20,20))
sns.heatmap(base_house.corr(), annot=True)
plt.show()

x= base_house.iloc[:,4:5].values
print(x)

y = base_house.iloc[:, 1].values
print(y)

from sklearn.model_selection import train_test_split
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)

print(X_casas_treinamento.shape, y_casas_treinamento.shape)

print(X_casas_teste.shape, y_casas_teste.shape)

from sklearn.linear_model import LinearRegression

regressor_simples=LinearRegression()
regressor_simples.fit(X_casas_treinamento, y_casas_treinamento)

print(regressor_simples.intercept_)#b0

print(regressor_simples.coef_)#b1

print(regressor_simples.score(X_casas_treinamento, y_casas_treinamento))

print(regressor_simples.score(X_casas_teste, y_casas_teste))

#desempenho bem ruim

previsoes=regressor_simples.predict(X_casas_treinamento)
grafico=px.scatter(x=X_casas_treinamento.ravel(), y=previsoes)


grafico1 = px.scatter(x = X_casas_treinamento.ravel(), y = y_casas_treinamento)
grafico2 = px.line(x = X_casas_treinamento.ravel(), y = previsoes)
grafico2.data[0].line.color = 'red'
grafico3 = go.Figure(data=grafico1.data + grafico2.data)
grafico3.show()


#calculando erro
previsoes_teste= regressor_simples.predict(X_casas_teste)
print(previsoes_teste)
print(y_casas_teste)

print((abs(y_casas_teste-previsoes_teste)).mean())##media de erro de 172 mil, bem alto

from sklearn.metrics import mean_absolute_error, mean_squared_error

abso=mean_absolute_error(y_casas_teste, previsoes_teste)##mesma coisa q ali em cima
print(abso)


grafico1 = px.scatter(x = X_casas_teste.ravel(), y = y_casas_teste)
grafico2 = px.line(x = X_casas_teste.ravel(), y = previsoes_teste)
grafico2.data[0].line.color = 'red'
grafico3 = go.Figure(data=grafico1.data + grafico2.data)
grafico3.show()
##comportamento bem parecido com a base de treinamento
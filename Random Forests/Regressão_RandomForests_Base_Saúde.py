import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor


base_saude2= pd.read_csv(r'C:\Users\joaov\Desktop\OneDrive\CSV\plano_saude2.csv')

x=base_saude2.iloc[:, 0:1].values
y=base_saude2.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_sa_treinamento, X_sa_teste, y_sa_treinamento, y_sa_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)

saude_random=RandomForestRegressor(n_estimators=100)
saude_random.fit(X_sa_treinamento, y_sa_treinamento)

score=saude_random.score(X_sa_teste, y_sa_teste)
print(score)

previsoes=saude_random.predict(X_sa_teste)
print(previsoes)
print(y_sa_teste)

from sklearn.metrics import mean_absolute_error
abso=mean_absolute_error(y_sa_teste, previsoes)
print(abso)


x_teste_arvore= np.arange(min(X_sa_treinamento), max(X_sa_treinamento), 0.1)
print(x_teste_arvore)
x_teste_arvore=x_teste_arvore.reshape(-1,1)

grafico = px.scatter(x = X_sa_treinamento.ravel(), y = y_sa_treinamento)
grafico.add_scatter(x = x_teste_arvore.ravel(), y = saude_random.predict(x_teste_arvore), name = 'Regressão')
grafico.show()

previsaoteste= saude_random.predict([[38]])
print(previsaoteste)
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


base_house= pd.read_csv(r'C:\Users\joaov\OneDrive\Área de Trabalho\I.A-Classificação\CSV\house_prices.csv')


x=base_house.iloc[:, 3:19].values


y=base_house.iloc[:, 2].values


from sklearn.model_selection import train_test_split
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)



regressor_multiplo=LinearRegression()

regressor_multiplo.fit(X_casas_treinamento, y_casas_treinamento)

print(regressor_multiplo.intercept_)

score=regressor_multiplo.score(X_casas_treinamento, y_casas_treinamento)
print(score)

score2=regressor_multiplo.score(X_casas_teste, y_casas_teste)
print(score2)

#melhoria muito grande em relação a regressão simples

previsoes=regressor_multiplo.predict(X_casas_teste)
print(previsoes)
print(y_casas_teste)


from sklearn.metrics import mean_absolute_error, mean_squared_error

abso=mean_absolute_error(y_casas_teste, previsoes)##mesma coisa q ali em cima
print(abso)

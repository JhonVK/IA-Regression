import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


base_house= pd.read_csv(r'C:\Users\joaov\OneDrive\Área de Trabalho\I.A-Classificação\CSV\house_prices.csv')


x=base_house.iloc[:, 3:19].values


y=base_house.iloc[:, 2].values


from sklearn.model_selection import train_test_split
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(x, y, test_size = 0.3, random_state = 0)

regressor_random_forest_casas= RandomForestRegressor(n_estimators=100)
regressor_random_forest_casas.fit(X_casas_treinamento, y_casas_treinamento)

score=regressor_random_forest_casas.score(X_casas_teste, y_casas_teste)
print(score)

previsoes=regressor_random_forest_casas.predict(X_casas_teste)
print(previsoes)

from sklearn.metrics import mean_absolute_error
abso=mean_absolute_error(y_casas_teste, previsoes)
print(abso)
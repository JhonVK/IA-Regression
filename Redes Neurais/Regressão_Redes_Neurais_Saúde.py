import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

base_saude2= pd.read_csv(r'C:\Users\joaov\Desktop\OneDrive\CSV\plano_saude2.csv')


x=base_saude2.iloc[:, 0:1].values
y=base_saude2.iloc[:, 1].values

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X_plano_saude2_scaled = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y_plano_saude2_scaled = scaler_y.fit_transform(y.reshape(-1,1))

print(X_plano_saude2_scaled)

print(y_plano_saude2_scaled)

from sklearn.neural_network import MLPRegressor
regressor_rn_saude=MLPRegressor(max_iter=1600)
regressor_rn_saude.fit(X_plano_saude2_scaled, y_plano_saude2_scaled.ravel())

score=regressor_rn_saude.score(X_plano_saude2_scaled, y_plano_saude2_scaled)
print(score)

grafico = px.scatter(x = X_plano_saude2_scaled.ravel(), y = y_plano_saude2_scaled.ravel())
grafico.add_scatter(x = X_plano_saude2_scaled.ravel(), y = regressor_rn_saude.predict(X_plano_saude2_scaled), name = 'Regressão')
grafico.show()
plt.show()

novo = [[40]]
novo = scaler_x.transform(novo)
print(novo)
predi=scaler_y.inverse_transform(regressor_rn_saude.predict(novo).reshape(-1, 1))
print(predi)

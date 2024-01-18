import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

base_saude2= pd.read_csv(r'C:\Users\joaov\OneDrive\Área de Trabalho\I.A-Classificação\CSV\plano_saude2.csv')


x=base_saude2.iloc[:, 0:1].values
y=base_saude2.iloc[:, 1].values

from sklearn.svm import SVR
#svm rbf precisa normalizar(os outros nao precisam pois normalizam internamente)

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X_plano_saude2_scaled = scaler_x.fit_transform(x)
scaler_y = StandardScaler()
y_plano_saude2_scaled = scaler_y.fit_transform(y.reshape(-1,1))

#kernel linear
regressor_svr= SVR(kernel='linear')
regressor_svr.fit(x, y)

grafico = px.scatter(x = x.ravel(), y = y)
grafico.add_scatter(x = x.ravel(), y = regressor_svr.predict(x), name = 'Regressão')
grafico.show()


#kernel Polinomial
regressor_svr= SVR(kernel='poly', degree=5)
regressor_svr.fit(x, y)

grafico = px.scatter(x = x.ravel(), y = y)
grafico.add_scatter(x = x.ravel(), y = regressor_svr.predict(x), name = 'Regressão')
grafico.show()

# Kernel rbf (sensivel a escala de dados cuidado)
regressor_svr_saude_rbf = SVR(kernel='rbf')
regressor_svr_saude_rbf.fit(X_plano_saude2_scaled, y_plano_saude2_scaled.ravel())

grafico = px.scatter(x = X_plano_saude2_scaled.ravel(), y = y_plano_saude2_scaled.ravel())
grafico.add_scatter(x = X_plano_saude2_scaled.ravel(), y = regressor_svr_saude_rbf.predict(X_plano_saude2_scaled), name = 'Regressão')
grafico.show()

novo=[[40]]
novo=scaler_x.transform(novo)
print(scaler_y.inverse_transform(regressor_svr_saude_rbf.predict(novo).reshape(-1,1)))

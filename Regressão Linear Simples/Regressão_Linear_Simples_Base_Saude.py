import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Base bem pequena, apenas para entender os conceitos/algoritmos

base_plano_saude= pd.read_csv(r'C:\Users\joaov\OneDrive\Área de Trabalho\I.A-Classificação\CSV\plano_saude.csv')
print(base_plano_saude)

x=base_plano_saude.iloc[:,0].values             #idade
print(x)
y=base_plano_saude.iloc[:,1].values            #custo do plano
print(y)


print(np.corrcoef(x, y))

print(x.shape)

x=x.reshape(-1,1)

print(x)
print(x.shape)

from sklearn.linear_model import LinearRegression
regressor_saude=LinearRegression()

regressor_saude.fit(x, y)

#b0
print(regressor_saude.intercept_)

#b1
print(regressor_saude.coef_)

previsoes=regressor_saude.predict(x)
print(previsoes)



grafico= px.scatter(x= x.ravel(), y= y)
grafico.add_scatter(x=x.ravel(), y= previsoes, name='regressao')
grafico.show()

#nota que nao é tao preciso pois é linear(uma reta)

print(regressor_saude.score(x, y))#score

from yellowbrick.regressor import ResidualsPlot

visualizador=ResidualsPlot(regressor_saude)
visualizador.fit(x, y)
visualizador.poof()
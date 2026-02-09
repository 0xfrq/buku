import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


lifesat = pd.read_csv("https://gitlab.au.dk/mal-grp22/mal-grp22/-/raw/bdd90f0791757c97e1c13683d20167bac331e7e3/datasets/lifesat/lifesat.csv")
x = lifesat[["GDP per capita"]].values
y = lifesat[["Life satisfaction"]].values

lifesat.plot(kind='scatter', grid=True, x="GDP per capita", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show

model = KNeighborsRegressor(n_neighbors=3)

model.fit(x,y)

x_new = [[37_655.2]]
print(model.predict(x_new))
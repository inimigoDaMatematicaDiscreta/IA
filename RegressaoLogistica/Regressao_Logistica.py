import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data2.txt", sep=",", header=None)
df.columns = ['exame_1', 'exame_2', 'classe']

ax = sns.scatterplot(x='exame_1', y='exame_2', hue='classe', data=df, style='classe', s=80)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[1:], ['Doente', 'Saudável'])
plt.title('Dados do treinamento')
plt.show() 

#não linear
X = df[["exame_1", "exame_2"]]
y = df["classe"]
X_poly = X.copy()
X_poly["exame_1^2"] = X["exame_1"]**2
X_poly["exame_2^2"] = X["exame_2"]**2
X_poly["exame_1*exame_2"] = X["exame_1"] * X["exame_2"]

#treina o modelo 
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_poly, y)

#cria o grid 
x1_min, x1_max = X["exame_1"].min() - 0.5, X["exame_1"].max() + 0.5
x2_min, x2_max = X["exame_2"].min() - 0.5, X["exame_2"].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                       np.linspace(x2_min, x2_max, 300))
grid = pd.DataFrame({
    "exame_1": xx1.ravel(),
    "exame_2": xx2.ravel()
})
grid["exame_1^2"] = grid["exame_1"]**2
grid["exame_2^2"] = grid["exame_2"]**2
grid["exame_1*exame_2"] = grid["exame_1"] * grid["exame_2"]

#prevê
pred = modelo.predict(grid)
pred = pred.reshape(xx1.shape)

#mostra
plt.figure(figsize=(8, 6))
plt.contourf(xx1, xx2, pred, alpha=0.3, cmap="coolwarm")
sns.scatterplot(x=X["exame_1"], y=X["exame_2"], hue=y, style=y, s=80, edgecolor="k")
plt.title("Superfície de decisão")
plt.xlabel("exame_1")
plt.ylabel("exame_2")
plt.show()

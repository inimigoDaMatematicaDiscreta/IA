import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

N = 10
x = np.linspace(0, 1, N)
y = np.sin(2*np.pi*x) + np.random.normal(0, 0.15, N)

NR = 100
xr = np.linspace(0, 1, NR)
yr = np.sin(2*np.pi*xr)

K = 10 #mude aqui

mymodel = np.poly1d(np.polyfit(x, y, K))


line1, = plt.plot(xr, mymodel(xr), label=f'Regressão com K ={K}')
line2, = plt.plot(xr, yr, label='Distribuição')

plt.scatter(x, y)
plt.legend(handles=[line1, line2], )
plt.show()
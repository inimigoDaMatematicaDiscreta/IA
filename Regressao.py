import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df2 = pd.read_csv('ex1data2.txt', sep=',', header=None)
df2.columns = ['house_size', 'bedrooms', 'house_price']


def feature_normalize(X, mean=np.zeros(1), std=np.zeros(1)):
    X = np.array(X)
    if len(mean.shape) == 1 or len(std.shape) == 1:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1)
    X = (X - mean)/std
    return X, mean, std


X_norm, mu, sigma = feature_normalize(df2[['house_size', 'bedrooms']])
m = df2.shape[0]
X = np.hstack((np.ones((m,1)), X_norm))  #Adiciona a coluna de váarios uns para theta_0
y = df2.house_price.values.reshape(-1, 1)
theta = np.zeros((X.shape[1], 1))


def compute_cost(X, y, theta):
    m = y.shape[0]
    h = X.dot(theta)
    J = (1/(2*m)) * ((h - y).T.dot(h - y))
    return J

def gradient_descent(X, y, theta, alpha, num_iters, E=1e-3):
    m = y.shape[0]
    J_history = np.zeros(shape=(num_iters, 1))

    for i in range(0, num_iters):
        h = X.dot(theta)
        diff_hy = h - y

        delta = (1/m) * (diff_hy.T.dot(X))
        theta = theta - (alpha * delta.T)
        
        J_history[i] = compute_cost(X, y, theta)

        #condição de convergência
        if i > 0 and abs(J_history[i] - J_history[i-1]) < E:
            print(f'Convergiu em {i} iterações com alpha = {alpha}')
            break

    return theta, J_history



alphas = [0.3, 0.1, 0.03, 0.01]
colors = ['b', 'r', 'g', 'c']
num_iters = 50

plt.figure(figsize=(10,6))
for i in range(len(alphas)):
    theta_init = np.zeros((X.shape[1], 1))
    theta_temp, J_hist = gradient_descent(X, y, theta_init, alphas[i], num_iters)
    plt.plot(range(len(J_hist)), J_hist, colors[i], label=f'alpha = {alphas[i]}')

plt.xlabel('Número de iterações')
plt.ylabel('Custo J(θ)')
plt.title('Custo J(θ) por iteração para diferentes valores de alpha')
plt.legend()
plt.grid(True)
plt.show()


alpha = 0.1
iterations = 250
theta = np.zeros((X.shape[1], 1))
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

print('\nTheta encontrado pela descida do gradiente:')
print(theta)


sqft = (1650 - mu[0]) / sigma[0]
bedrooms = (3 - mu[1]) / sigma[1]
y_pred = theta[0] + theta[1]*sqft + theta[2]*bedrooms
print(f'\nPreço previsto para casa de 1650m² e 3 quartos: ${y_pred[0]:.2f}')

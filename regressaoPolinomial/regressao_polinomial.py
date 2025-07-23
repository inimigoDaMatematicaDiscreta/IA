import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

def cost_function(theta, X, y, lambda_=0.1):

    m = y.size
    h = np.dot(X, theta)
    alpha = h - y

    #Cálculo custo com regl 
    J = (1/(2*m)) * np.sum(alpha**2) + (lambda_/(2*m)) * np.sum(theta[1:]**2)
    
    #Cálculo do gradiente
    grad = np.zeros_like(theta)
    grad[0] = (1/m) * np.dot(X[:, 0], alpha)  #Sem regl do theta0 (bias)
    grad[1:] = (1/m) * np.dot(X[:, 1:].T, alpha) + (lambda_/m) * theta[1:]  #Com regl dos thetas (MSE + regl)
    
    return J, grad

def optimize_theta(X, y, initial_theta, lambda_=0.1):
    opt_results = opt.minimize(cost_function, initial_theta, args=(X, y, lambda_), method='L-BFGS-B',
                               jac=True, options={'maxiter': 400})
    if not opt_results.success:
        raise RuntimeError("Otimização falhou: " + opt_results.message)
    return opt_results['x'], opt_results['fun']

def feature_normalize(X, mean=None, std=None):
    X = np.array(X)
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1)
    X = (X - mean) / std
    return X, mean, std

def extend_feature(X_ini, k):
    result = X_ini
    for i in range(2, k+1):
        result = np.hstack((result, np.power(X_ini, i)))
    return result

# Geração de dados
N = 30
x = np.linspace(0, 1, N)
y = np.sin(2*np.pi*x) + np.random.normal(0, 0.5, N)

NR = 100
xr = np.linspace(0, 1, NR)
yr = np.sin(2*np.pi*xr)

m = y.size

# Preparação dos dados
k = 9
X_ini = x.copy()
X_ini = X_ini.reshape(-1, 1)
X = extend_feature(X_ini, k)
X, mean, std = feature_normalize(X)
ones = np.ones((m, 1))
print(X.shape)
print(ones.shape)
X = np.hstack([ones, X])

#X = np.hstack([np.ones((m, 1)), X])  # Correção aplicada aqui
theta = np.random.randn(k + 1)

# Otimização
opt_theta, cost = optimize_theta(X, y, theta)

# Previsão
xnew = np.linspace(0, 1, 50)
xnew = xnew.reshape(-1, 1)
X2 = extend_feature(xnew, k)
X2 = (X2 - mean) / std
X2 = np.hstack([np.ones((xnew.shape[0], 1)), X2])
h = np.dot(X2, opt_theta)

# Visualização
line1, = plt.plot(xnew, h, label='Regression')
line2, = plt.plot(xr, yr, label='True distribution')
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression of Order 9')
plt.legend(handles=[line1, line2])
plt.show()
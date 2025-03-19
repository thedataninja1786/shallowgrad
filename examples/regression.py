from shallowgrad.nn import nn
from optimizers.optimizers import Adam
import numpy as np
from sklearn.metrics import mean_squared_error

np.random.seed(0)
num_samples = 2000
num_features = 5

X = np.random.rand(num_samples, num_features)
noise = np.random.randn(num_samples, 1) * 0.1
true_coeffs = np.array([[2.0], [-1.0], [0.5], [1.5], [-0.5]])
true_intercept = 1.0
Y = np.dot(X, true_coeffs) + noise

l1 = nn.Linear(5, 1000, activation="LeakyReLU", bias=True)
l2 = nn.Linear(1000, 500, activation="LeakyReLU", bias=True)
l3 = nn.Linear(500, 1, bias=True)
loss = nn.MeanSquaredLoss()
optim = Adam(layers=[l1, l2, l3], lr=1e-4)

BS = 512
for _ in range(200):
    preds = []
    samp = np.random.randint(0, X.shape[0], size=(BS))
    x = X[samp]
    y = Y[samp]
    x = l1(x)
    x = l2(x)
    x = l3(x)

    preds.append(x)

    l = loss(x, y)
    loss.backwards()
    optim.step()

print(mean_squared_error(y, np.array(preds).reshape(-1, 1)))

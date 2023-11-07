from shallowgrad.nn import nn 
from optimizers.optimizers import Adam
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


np.random.seed(0)
num_samples = 2000
num_features = 5  # Number of features
X_i = np.random.rand(num_samples, num_features)  # Random input features
noise = np.random.randn(num_samples, 1) * 0.1  # Random noise
true_coeffs = np.array([[2.0], [-1.0], [0.5], [1.5], [-0.5]])  # True coefficients
true_intercept = 1.0
Y = np.dot(X_i,true_coeffs) + noise
X = np.array(X_i)

l1 = nn.Linear(5,1000,activation='LeakyReLU',bias=False)
l2 = nn.Linear(1000,500,activation='LeakyReLU',bias=False)
l3 = nn.Linear(500,1,bias=False)
loss = nn.MeanSquaredLoss()
optim = Adam(layers=[l1,l2,l3],lr=1e-4)

BS = 512
for i in range(500):
  preds = []
  samp = np.random.randint(0, X.shape[0], size=(BS))
  x = X[samp]
  y = Y[samp]
  x = l1(x)
  #x = nn.LeakyReLU()(x)
  x = l2(x)
  #x = nn.LeakyReLU()(x)
  x = l3(x)

  for el in x:
    preds.append(el[0])

  l = loss(x,y)
  loss.backwards()
  optim.step()

  if i % 10 == 0: print(mean_squared_error(y,preds))
print('R_squared: ',r2_score(preds,y))
#print(plt.scatter(y,preds))
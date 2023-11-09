import numpy as np
class Adam:
  def __init__(self,layers=[],lr=0.001,b1=0.9,b2=0.999,e=1e-10):
    self.layers = layers
    self.lr = lr
    self.b1 = b1 
    self.b2 = b2
    self.e = e
    self.t = 0 

    self.m = [np.zeros_like(l.weights) for l in self.layers]
    self.v = [np.zeros_like(l.weights) for l in self.layers]

  def step(self):
    for i,l in enumerate(self.layers):
      self.t += 1
      self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * l.grad
      self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(l.grad)
      m_hat = self.m[i] / (1 - self.b1 ** self.t)
      v_hat = self.v[i] / (1 - self.b2 ** self.t)
      l.weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.e)

class SGD:
  def __init__(self,layers=[],lr=0.001):
    self.layers = layers
    self.lr = lr

  def step(self):
    for l in self.layers:
      l.weights -= self.lr * l.grad

class RMSprop:
  def __init__(self,layers=[],lr=0.001,beta=0.9,e=1e-10):
    self.layers = layers
    self.lr = lr
    self.beta = beta
    self.e = e

    self.v = [np.zeros_like(l.weights) for l in self.layers]

  def step(self):
    for i,l in enumerate(self.layers):
      self.v[i] = self.beta * self.v[i] + (1 - self.beta) * np.square(l.grad)
      l.weights -= self.lr * l.grad / (np.sqrt(self.v[i]) + self.e)
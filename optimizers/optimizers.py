import numpy as np
class Adam:
  def __init__(self,layers=[],lr=0.001,b1=0.9,b2=0.999,e=1e-8):
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
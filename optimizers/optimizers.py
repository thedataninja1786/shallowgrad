import numpy as np


class Chris:
    def __init__(self, layers=[], lr=0.001, b1=0.9, b2=0.999, e=1e-10):
        self.layers = layers
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.t = 0

        self.m = [np.zeros_like(l.weights) for l in self.layers]
        self.v = [np.zeros_like(l.weights) for l in self.layers]
        self.m_bias = [np.zeros_like(l.bias) for l in self.layers if l.use_bias]
        self.v_bias = [np.zeros_like(l.bias) for l in self.layers if l.use_bias]

    def step(self):
        for i, l in enumerate(self.layers):
            self.t += 1
            self.b1 * self.m[i] + (1 - self.b1) * (
                np.abs(l.grad - np.median(l.grad, axis=0))
            )
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(
                l.grad - np.median(l.grad, axis=0)
            )
            m_hat = self.m[i]  # / (1 - self.b1 ** self.t)
            v_hat = self.v[i]  # / (1 - self.b2 ** self.t)
            l.weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.e)

            if l.use_bias:
                self.m_bias[i] = self.b1 * self.m_bias[i] + (1 - self.b1) * l.grad_bias
                self.v_bias[i] = self.b2 * self.v_bias[i] + (1 - self.b2) * np.square(
                    l.grad_bias
                )
                m_hat_bias = self.m_bias[i] / (1 - self.b1**self.t)
                v_hat_bias = self.v_bias[i] / (1 - self.b2**self.t)
                l.bias -= self.lr * m_hat_bias / (v_hat_bias + self.e)


class Adam:
    def __init__(self, layers=[], lr=0.001, b1=0.9, b2=0.999, e=1e-10):
        self.layers = layers
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.t = 0

        self.m = [np.zeros_like(l.weights) for l in self.layers]
        self.v = [np.zeros_like(l.weights) for l in self.layers]
        self.m_bias = [np.zeros_like(l.bias) for l in self.layers if l.use_bias]
        self.v_bias = [np.zeros_like(l.bias) for l in self.layers if l.use_bias]

    def step(self):
        for i, l in enumerate(self.layers):
            self.t += 1
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * l.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(l.grad)
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            l.weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.e)

            if l.use_bias:
                self.m_bias[i] = self.b1 * self.m_bias[i] + (1 - self.b1) * l.grad_bias
                self.v_bias[i] = self.b2 * self.v_bias[i] + (1 - self.b2) * np.square(
                    l.grad_bias
                )
                m_hat_bias = self.m_bias[i] / (1 - self.b1**self.t)
                v_hat_bias = self.v_bias[i] / (1 - self.b2**self.t)
                l.bias -= self.lr * m_hat_bias / (np.sqrt(v_hat_bias) + self.e)


class SGD:
    def __init__(self, layers=[], lr=0.001):
        self.layers = layers
        self.lr = lr

    def step(self):
        for l in self.layers:
            l.weights -= self.lr * l.grad
            if l.use_bias:
                l.bias -= self.lr * l.grad_bias


class RMSprop:
    def __init__(self, layers=[], lr=0.001, beta=0.9, e=1e-10):
        self.layers = layers
        self.lr = lr
        self.beta = beta
        self.e = e

        self.v = [np.zeros_like(l.weights) for l in self.layers]
        self.v_bias = [np.zeros_like(l.bias) for l in self.layers if l.use_bias]

    def step(self):
        for i, l in enumerate(self.layers):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * np.square(l.grad)
            self.v_bias[i] = self.beta * self.v_bias[i] + (1 - self.beta) * np.square(
                l.grad_bias
            )
            l.weights -= self.lr * l.grad / (np.sqrt(self.v[i]) + self.e)
            if l.use_bias:
                l.bias -= self.lr * l.grad_bias / (np.sqrt(self.v_bias[i]) + self.e)


class Adagrad:
    def __init__(self, layers=[], lr=0.001, e=1e-10):
        self.layers = layers
        self.lr = lr
        self.e = e

        self.v = [np.zeros_like(l.weights) for l in self.layers]
        self.v_bias = [np.zeros_like(l.bias) for l in self.layers if l.use_bias]

    def step(self):
        for i, l in enumerate(self.layers):
            self.v[i] += np.square(l.grad)
            self.v_bias[i] += np.square(l.grad_bias)
            l.weights -= self.lr * l.grad / (np.sqrt(self.v[i]) + self.e)
            if l.use_bias:
                l.bias -= self.lr * l.grad_bias / (np.sqrt(self.v_bias[i]) + self.e)

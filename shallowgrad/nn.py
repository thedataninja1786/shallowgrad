import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    def __repr__(self):
        pass

    def backward(self, x):
        pass


class nn:

    class ActivationMapping:

        @staticmethod
        def get_mapping():
            return {
                "ReLU": nn.ReLU(),
                "LeakyReLU": nn.LeakyReLU(),
                "Tanh": nn.Tanh(),
                "Sigmoid": nn.Sigmoid(),
                "Softmax": nn.Softmax(),
            }

    class loss:
        instances = []

        @staticmethod
        def backward_prop(g_loss):  # g of loss
            delta = np.copy(g_loss)
            for i in reversed(range(len(nn.loss.instances))):
                delta = nn.loss.instances[i].backward(delta)

    # Activation functions
    class Softmax(Activation):
        def __call__(self, x):
            x = x - np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(x)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

        def __repr__(self):
            return "Softmax applied in place"

        def backward(self, x):
            return x * (1 - x)

    class ReLU(Activation):
        def __call__(self, x):
            return np.maximum(0, x)

        def __repr__(self):
            return "ReLU applied in place"

        def backward(self, x):
            return (x > 0).astype(int)

    class LeakyReLU(Activation):
        def __call__(self, x, a=0.5):
            return np.maximum(a * x, x)

        def __repr__(self):
            return "LeakyReLU applied in place"

        def backward(self, x, a=0.5):
            return (x > 0).astype(int) + (a * (x <= 0)).astype(int)

    class Tanh(Activation):
        def __call__(self, x):
            return np.tanh(x)

        def __repr__(self):
            return "Tanh applied in place"

        def backward(self, x):
            return 1 - np.tanh(x) ** 2

    class Sigmoid(Activation):
        def __call__(self, x):
            if x >= 0:
                return 1 / (1 + np.exp(-x))
            else:
                return np.exp(x) / (1 + np.exp(x))

        def __repr__(self):
            return "Sigmoid applied in place"

        def backward(self, x):
            return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))

    # Loss functions
    class MeanSquaredLoss(loss):
        def __init__(self):
            self.y_pred = None
            self.y_true = None
            super().__init__()

        def __call__(self, y_pred, y_true):
            self.y_pred = y_pred
            self.y_true = y_true
            return ((self.y_true - self.y_pred) ** 2).mean()

        def backwards(self):
            g = 2 * (self.y_pred - self.y_true) / len(self.y_true)
            self.backward_prop(g)

    class CrossEntropyLoss(loss):
        def __init__(self):
            self.y_pred = None
            self.y_true = None
            self.e = 1e-10
            super().__init__()

        def __call__(self, y_pred, y_true):
            self.y_true = y_true

            if len(y_true.shape) == 1 or y_true.shape[1] == 1:  # convert to one-hot
                num_classes = np.max(y_true) + 1
                one_hot = np.eye(num_classes)[y_true.flatten()]
                one_hot = one_hot.reshape(-1, num_classes)
                self.y_true = one_hot

            self.y_pred = nn.Softmax()(y_pred)
            self.y_pred = np.clip(self.y_pred, self.e, 1 - self.e)
            return np.mean(-np.sum(self.y_true * np.log(self.y_pred + self.e), axis=1))

        def backwards(self):
            g = (self.y_pred - self.y_true) / self.y_true.shape[0]
            self.backward_prop(g)

    class BinaryCrossEntropyLoss(loss):
        def __init__(self):
            self.y_pred = None
            self.y_true = None
            super().__init__()
            self.e = 1e-15

        def __call__(self, y_pred, y_true):
            self.y_pred = y_pred
            self.y_true = y_true
            return -np.mean(
                (self.y_true * np.log(self.y_pred + self.e))
                + ((1 - self.y_true) * np.log(1 - self.y_pred + self.e))
            )

        def backwards(self):
            g = (self.y_pred - self.y_true) / (
                (self.y_pred + self.e) * (1 - self.y_pred + self.e)
            )
            self.backward_prop(g)

    class Linear:
        def __init__(self, in_features, out_features, bias=True, activation=None):
            self.in_features = in_features
            self.out_features = out_features
            self.use_bias = bias
            self.bias = None
            self.activation_func = activation
            self.weights = self._weight_init()
            self.grad = None
            self.forward_pass = None
            self.x = None
            self.grad_bias = None
            self.activation_mapping = nn.ActivationMapping.get_mapping()
            nn.loss.instances.append(self)  # record instance for backprop

            if activation in self.activation_mapping:
                self.activation_func = self.activation_mapping.get(activation, None)

        def __call__(self, x):  # forward
            self.x = x
            if self.activation_func is not None:
                self.forward_pass = self.activation_func(
                    (self.x.dot(self.weights) + self.bias)
                )
            else:
                self.forward_pass = self.x.dot(self.weights) + self.bias
            return self.forward_pass

        def __repr__(self):
            return f"Linear(in_features={self.in_features}, out_features={self.out_features}, activation={self.activation_func})"

        def _weight_init(self):
            """Glorot uniform initialization"""
            self.bias = (
                np.random.uniform(size=(1, self.out_features)) * 0.05
                if self.bias
                else 0
            )
            v = np.sqrt(2.0 / (self.in_features + self.out_features))
            return np.random.normal(0, v, size=(self.in_features, self.out_features))

        def backward(self, delta):
            if self.activation_func is not None:
                # Calculate gradient of the activation function
                delta = delta * self.activation_func.backward(self.forward_pass)
            # Compute gradients for weights and bias
            self.grad = np.dot(self.x.T, delta)
            if self.use_bias:
                self.grad_bias = np.sum(delta, axis=0, keepdims=True)
            # Compute gradient with respect to x
            d_x = np.dot(delta, self.weights.T)
            return d_x

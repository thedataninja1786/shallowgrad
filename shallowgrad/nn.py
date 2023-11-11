import numpy as np
# TO DO (before moving to CNN)
  # implement all loss functions binary/categorical cross-entropy 
  # utilize and update (estimate gradient of) bias -> done
  # Implement Batch/Instance Norm -> done
  # Add gradient of softmax -> done
  # fix backprobagation
  # Add optimizer (Adam) handles lr and weight-update
  # add tests
  # add glorot uniform weight init -> done

class nn:
  class loss:
    instances = []
 
    @staticmethod
    def backward_prop(g_loss):
        delta = np.copy(g_loss)
        for i in reversed(range(len(nn.loss.instances))):
            delta = nn.loss.instances[i].backward(delta)

  class Softmax:
    def __init__(self):
      pass

    def __call__(self,x):
      return self.softmax(x)

    def softmax(self,x):
      x = x - np.max(x, axis=1, keepdims=True)
      exp_x = np.exp(x)
      softmax_output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
      self.soft_out = softmax_output
      return softmax_output
    
    @staticmethod
    def backward(x):
      return x * (1 - x)


  class ReLU:
    def __init__(self):
      self.x = None

    def __call__(self,x):
      return self.relu(x)

    def relu(self,x):
      self.x = x
      return np.maximum(0,self.x)
    
    def __repr__(self):
      return f"ReLU applied in place with output shape: {x.shape}"

    @staticmethod
    def backward(x): 
      return (x > 0).astype(int)


  class LeakyReLU:
    def __init__(self):
      self.x = None

    def leakyrelu(self,x,a=0.5):
      self.x = x
      return np.maximum(a * x, x)
 
    def __call__(self,x,a=0.5):
      return self.leakyrelu(x,a)

    def backward(self,x,a=0.5):
      return (x > 0).astype(int) + (a * (x <= 0)).astype(int)
    

  class Tanh:
    @staticmethod
    def __call__(x):
      return np.tanh(x)
    
    @staticmethod
    def tanh(x):
      return np.tanh(x)
      
    @staticmethod
    def backward(x):
      return 1 - np.tanh(x) ** 2
    
  class Sigmoid:
    def __init__(self):
      self.x = None 

    def __call__(self,x):
      return self.sigmoid(x)

    def sigmoid(self,x):
      self.x = x
      if x >= 0:
        return 1 / (1 + np.exp(-x))
      else:
        return np.exp(x) / (1 + np.exp(x)) 
    
    def backward(self,x):
      return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))



  class MeanSquaredLoss(loss):
    def __init__(self):
      self.y_pred = None
      self.y_true = None
      super().__init__()

    def __call__(self,y_pred,y_true):
      self.y_pred = y_pred; self.y_true = y_true
      return ((self.y_true - self.y_pred) ** 2).mean()

    def backwards(self):
      g = 2 * (self.y_pred - self.y_true) / len(self.y_true) # grad of loss function
      self.backward_prop(g)

  class CrossEntropyLoss(loss): #compare this tomorrow with pytorch and see if they return the same
    def __init__(self):
      self.y_pred = None
      self.y_true = None
      self.e = 1e-10
      super().__init__()

    def __call__(self, y_pred, y_true):
      self.y_true = y_true
      
      if len(y_true.shape) == 1 or y_true.shape[1] == 1: # convert to one-hot
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

    def __call__(self,y_pred,y_true):
      self.y_pred = y_pred; self.y_true = y_true
      return -np.mean((self.y_true * np.log(self.y_pred + self.e)) \
                      + ((1 - self.y_true) * np.log(1 - self.y_pred + self.e)))

    def backwards(self):
      g = (self.y_pred - self.y_true) / ((self.y_pred + self.e) * (1 - self.y_pred + self.e))
      self.backward_prop(g)

  class Linear:
    def __init__(self,in_features,out_features,bias=True,activation=None):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_func = activation
        self.weights = self._weight_init()
        self.grad = None
        self.forward_pass = None
        self.x = None
        self.grad_bias = None
        nn.loss.instances.append(self)
        super().__init__()

        if activation == 'ReLU':
            self.activation_func = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation_func = nn.LeakyReLU()
        elif activation == 'Tanh':
            self.activation_func = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation_func = nn.Sigmoid()
        elif activation == 'Softmax':
            self.activation_func = nn.Softmax()
        else: self.activation_func = None

    def __call__(self,x): # forward
        self.x = x
        if self.activation_func is not None:
            self.forward_pass = self.activation_func(self.x.dot(self.weights) + self.bias)
        else:
            self.forward_pass = self.x.dot(self.weights) + self.bias
        return self.forward_pass
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, activation={self.activation_func})"

    def _weight_init(self):
        """Initialize weights using Glorot uniform initialization"""
        self.bias = np.random.uniform(size=(1,self.out_features)) * 0.01 if self.bias else 0
        v = np.sqrt(2.0 / (self.in_features + self.out_features))
        return np.random.normal(0, v, size=(self.in_features, self.out_features)) 


    def backward(self, delta):
        if self.activation_func:
            # Calculate gradient of the activation function
            delta = delta * self.activation_func.backward(self.forward_pass)
        # Compute gradients for weights and bias
        self.grad = np.dot(self.x.T, delta)
        if self.bias is not None:
            self.grad_bias = np.sum(delta, axis=0, keepdims=True)
        # Compute gradient with respect to x
        d_x = np.dot(delta, self.weights.T)
        return d_x
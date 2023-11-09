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
    forward_passes = []
    activations = []
    lr = 1e-4
 
    @staticmethod
    def backward_prop(loss):
      #print('loss_shape',loss.shape,' ','output_layer_shape: ',nn.loss.forward_passes[-1].shape)
      delta = loss * nn.loss.instances[-1]._gradient(nn.loss.forward_passes[-1])
      #print('delta',delta.shape)
      for i in reversed(range(len(nn.loss.instances))):
        #print('forward_pass: ',nn.loss.forward_passes[i].shape)
        g = np.dot(nn.loss.forward_passes[i].T, delta)
        nn.loss.instances[i].grad = g
        
        #nn.loss.instances[i].weights -= g * nn.loss.lr
        #if nn.loss.instances[i].bias: nn.loss.instances[i].bias -= np.sum(delta) * (nn.loss.lr / 10)
        delta = np.dot(delta, nn.loss.instances[i].weights.T) * nn.loss.instances[i]._gradient(nn.loss.forward_passes[i])
        #print('transposed_shape_of_weights_after_update:', nn.loss.instances[i].weights.T.shape)
        
        #print('new_delta',delta.shape)

      nn.loss.forward_passes = [] #empty forward passes
    
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
    def g_softmax(x):
      return x * (1 - x)


  class ReLU:
    def __init__(self):
      self.x = None
      nn.loss.activations.append(self)

    def __call__(self,x):
      return self.relu(x)

    def relu(self,x):
      self.x = x
      return np.maximum(0,self.x)
    
    def __repr__(self):
      return f"ReLU applied in place with output shape: {x.shape}"

    @staticmethod
    def g_relu(x): 
      return (x > 0).astype(int)


  class LeakyReLU:
    def __init__(self):
      self.x = None

    def leakyrelu(self,x,a=0.5):
      self.x = x
      return np.maximum(a * x, x)
 
    def __call__(self,x,a=0.5):
      return self.leakyrelu(x,a)

    def g_leakyrelu(self,x,a=0.5):
      return (x > 0).astype(int) + (a * (x <= 0)).astype(int)
    

  class Tanh:
    @staticmethod
    def __call__(x):
      return np.tanh(x)
    
    @staticmethod
    def tanh(x):
      return np.tanh(x)
      
    @staticmethod
    def g_tanh(x):
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
    
    def g_sigmoid(self,x):
      return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))



  class MeanSquaredLoss(loss):
    def __init__(self):
      self.y_pred = None
      self.y_true = None
      super().__init__()

    def __call__(self,y_pred,y_true):
      self.y_pred = y_pred; self.y_true = y_true
      return np.mean(np.square((self.y_pred - self.y_true))) # mean ensures a scalar

    def backwards(self):
      g = self.y_pred - self.y_true # grad of loss function
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
        one_hot = np.zeros((y_true.size, y_pred.shape[1]))
        one_hot[np.arange(y_true.size), y_true] = 1
        self.y_true = one_hot
      
      self.y_pred = nn.Softmax()(y_pred)
      self.y_pred = np.clip(self.y_pred, self.e, 1 - self.e)
      return np.mean(-np.sum(self.y_true * np.log(self.y_pred + self.e), axis=1))

    def backwards(self):
      # gradient
      g = (self.y_pred - self.y_true) 
      self.backward_prop(g)

  class BinaryCrossEntropyLoss(loss):
    def __init__(self):
      self.y_pred = None
      self.y_true = None
      super().__init__()
      self.e = 1e-15 # prevent log(0)

    def __call__(self,y_pred,y_true):
      self.y_pred = y_pred; self.y_true = y_true
      return -np.mean((self.y_true * np.log(self.y_pred + self.e)) \
                      + ((1 - self.y_true) * np.log(1 - self.y_pred + self.e)))

    def backwards(self):
      # gradient
      g = (self.y_pred - self.y_true) / ((self.y_pred + self.e) * (1 - self.y_pred + self.e))
      self.backward_prop(g)

  class Linear(Softmax,ReLU,Tanh,Sigmoid,LeakyReLU):
    def __init__(self,in_features,out_features,bias=True,activation=None):
      self.in_features = in_features
      self.out_features = out_features
      self.bias = bias
      self.activation_func = activation
      self.weights = self._weight_init()
      self.grad = self._gradient(self.weights)
      nn.loss.instances.append(self)
      super().__init__()

    def __call__(self,x): # forward
      if not nn.loss.forward_passes: nn.loss.forward_passes.append(x)
      forward_pass = self._activation(x.dot(self.weights) + self.bias)
      nn.loss.forward_passes.append(forward_pass)
      return forward_pass
    
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        self._weights = new_weights
        self.grad = self._gradient(new_weights)
    
    def __repr__(self):
      return f"Linear(in_features={self.in_features}, out_features={self.out_features}, activation={self.activation_func})"

    def _weight_init(self):
      self.bias = np.random.uniform() * 0.01 if self.bias else 0
      limit = np.sqrt(6.0 / (self.in_features + self.out_features))
      return np.random.uniform(-1, 1, size=(self.in_features, self.out_features)) / np.sqrt(self.in_features * self.out_features)
    
    def _activation(self,x):
      if not self.activation_func:
        return x
      elif self.activation_func == 'ReLU':
        return self.relu(x)
      elif self.activation_func == 'LeakyReLU':
        return self.leakyrelu(x)
      elif self.activation_func == 'Tanh':
        return self.tanh(x)
      elif self.activation_func == 'Sigmoid':
        return self.sigmoid(x)
      elif self.activation_func == 'Softmax':
        return self.softmax(x)

    def _gradient(self,x):
      if not self.activation_func: return np.ones([x.shape[0],x.shape[1]])
      if self.activation_func == 'ReLU':
        return self.g_relu(x)
      elif self.activation_func == 'Tanh':
        return self.g_tanh(x)
      elif self.activation_func == 'Sigmoid':
        return self.g_sigmoid(x)
      elif self.activation_func == 'LeakyReLU':
        return self.g_leakyrelu(x)
      elif self.activation_func == 'Softmax':
        return self.g_softmax(x)
      else:
        raise ValueError(f"Cannot compute gradient for object {x}!")
<b>shallowgrad:<b> An oversimplified version of <a href https://pytorch.org/> PyTorch</a> based on numpy made for educational purposes.

It's extreme simplicity aims to dimistify the abstraction of modern deep learning frameworks while providing a similar interface to PyTorch.

<p align="center">
  <img src="shallowgrad.jpg" height ="350" width="350" title="hover text">
</p>

# MNIST in shallowgrad
<pre>
<code>
import numpy as np
from shallowgrad.nn import nn
from optimizers.optimizers import Adam 

# load MNIST
# ...

X_train = X_train.reshape((-1,784))
Y = Y_train.reshape(-1,1)
X = np.array(X_train / 255)

l1 = nn.Linear(784,2500,activation='ReLU',bias=True)
l2 = nn.Linear(2500,1000,activation='ReLU',bias=True)
l3 = nn.Linear(1000,10,bias=True) 
loss = nn.CrossEntropyLoss() 
optim = Adam(layers=[l1,l2,l3],lr=5-5)

y_hat = []
y_true = []
BS = 256
NUM_EPOCHS = 100

# training loop
for _ in range(NUM_EPOCHS):
    preds = []
    samp = np.random.randint(0, X.shape[0], size=(BS))
    x = X[samp]
    y = Y[samp]
    x = l1(x)
    x = l2(x)
    x = l3(x)
    preds_batch = np.argmax(x, axis=1).reshape(-1, 1)

    # Append batch predictions and true labels 
    y_hat.append(preds_batch)
    y_true.append(y)

    l = loss(x, y)
    loss.backwards()
    optim.step()
</code>
</pre>

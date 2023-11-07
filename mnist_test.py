import requests
import gzip
import os
import hashlib
import numpy as np
from nn import nn
from optimizers.optimizers import Adam 
import tempfile

home_dir = os.path.expanduser("~")
temp_dir = os.path.join(home_dir, "my_temp_directory")
os.makedirs(temp_dir, exist_ok=True)
"""
def fetch(url):

    fp = os.path.join(temp_dir, hashlib.md5(url.encode('utf-8')).hexdigest())
  
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
        
    #print(f"Downloaded data length: {len(dat)}")
    print('full_path',fp)
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 784))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:].reshape(-1,1)
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28 * 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:].reshape(-1,1)
"""

def read_file(fp):
    with open(fp, "rb") as f:
        dat = f.read()
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
X_train = read_file(r"C:\Users\cs2291\my_temp_directory\b0cdab8e37ae7c1c5560ee858afaac1d")[0x10:]
Y_train = read_file(r"C:\Users\cs2291\my_temp_directory\d4fdde61aca9f72d5fe2315410bb46a5")[8:]

X_train = X_train.reshape((-1,784))
Y = Y_train.reshape(-1,1)
X = np.array(X_train / 255)


l1 = nn.Linear(784,2500,activation='ReLU',bias=False)
l2 = nn.Linear(2500,1000,activation='ReLU',bias=False)
l3 = nn.Linear(1000,10,bias=False) # fix output shape of softmax
loss = nn.CrossEntropyLoss() # check if this works
optim = Adam(layers=[l1,l2,l3],lr=5-5)

all_preds = []
all_true_labels = []
BS = 256

for _ in range(200):
    preds = []
    samp = np.random.randint(0, X.shape[0], size=(BS))
    x = X[samp]
    y = Y[samp]
    x = l1(x)
    x = l2(x)
    x = l3(x)
    preds_batch = np.argmax(x, axis=1).reshape(-1, 1)

    # Append batch predictions and true labels to the lists
    all_preds.append(preds_batch)
    all_true_labels.append(y)

    l = loss(x, y)
    loss.backwards()
    optim.step()

# Concatenate predictions and true labels for all batches
all_preds = np.concatenate(all_preds, axis=0)
all_true_labels = np.concatenate(all_true_labels, axis=0)

print("Accuracy:", (np.array(all_preds) == all_true_labels).astype(int).mean())
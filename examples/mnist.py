from shallowgrad.nn import nn
from optimizers.optimizers import Adam
import gzip
import numpy as np


def read_file(fp):
    with open(fp, "rb") as f:
        dat = f.read()
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


X_train = read_file(r"datasets\b0cdab8e37ae7c1c5560ee858afaac1d")[0x10:]
Y_train = read_file(r"datasets\d4fdde61aca9f72d5fe2315410bb46a5")[8:]

X_train = X_train.reshape((-1, 784))
Y = Y_train.reshape(-1, 1)
X = np.array(X_train / 255)


l1 = nn.Linear(784, 2500, activation="ReLU", bias=True)
l2 = nn.Linear(2500, 1000, activation="ReLU", bias=True)
l3 = nn.Linear(1000, 10, bias=True)
loss = nn.CrossEntropyLoss()
optim = Adam(layers=[l1, l2, l3], lr=3e-4)

all_preds = []
all_true_labels = []
BS = 256

# training loop
for _ in range(200):
    preds = []
    samp = np.random.randint(0, X.shape[0], size=(BS))
    x = X[samp]
    y = Y[samp]
    x = l1(x)
    x = l2(x)
    x = l3(x)
    preds_batch = np.argmax(x, axis=1).reshape(-1, 1)

    # Append batch predictions and true labels for each batch
    all_preds.append(preds_batch)
    all_true_labels.append(y)

    l = loss(x, y)
    loss.backwards()
    optim.step()
    gradcheck(loss, x, y)

# Concatenate predictions and true labels for all batches
all_preds = np.concatenate(all_preds, axis=0)
all_true_labels = np.concatenate(all_true_labels, axis=0)

print("Accuracy:", (np.array(all_preds) == all_true_labels).astype(int).mean())

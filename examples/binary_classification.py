from shallowgrad.nn import nn
from optimizers.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv("datasets/heart.csv")
Y = df["target"].to_numpy()
del df["target"]
X = df.to_numpy()

l1 = nn.Linear(13, 250, activation="ReLU", bias=True)
l2 = nn.Linear(250, 100, activation="ReLU", bias=True)
l3 = nn.Linear(100, 1, activation="Sigmoid", bias=True)
loss = nn.BinaryCrossEntropyLoss()
optim = Adam(layers=[l1, l2, l3], lr=1e-4)

for _ in range(10):
    preds = []
    for x, y in zip(X, Y):
        x = x.reshape(1, -1)
        y = y.reshape(1, 1)
        x = l1(x)
        x = l2(x)
        x = l3(x)
        if x[0][0] <= 0.5:
            x[0] = 0
        else:
            x[0] = 1
        preds.append(x[0])

        l = loss(x, y)
        loss.backwards()
        optim.step()

print(accuracy_score(Y, preds))

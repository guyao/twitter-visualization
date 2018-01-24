import numpy as np

path = "data/"

X = np.loadtxt(path + "low_dim_embedding")

with open(path + "labels", encoding="utf-8") as f:
    lines = f.readlines()

labels = [label.strip() for label in lines]
label_vector = {}
for i, label in enumerate(labels):
    label_vector[label] = X[i, :]

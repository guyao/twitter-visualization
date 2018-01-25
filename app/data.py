import numpy as np
import os
from app import app

DATA_PATH = app.config["DATA_PATH"]
EMBEDDING_FILENAME = app.config["EMBEDDING_FILENAME"]
LOW_DIM_EMBEDDING_FILENAME = app.config["LOW_DIM_EMBEDDING_FILENAME"]
LABEL_FILENAME = app.config["LABEL_FILENAME"]

embedding = np.loadtxt(os.path.join(DATA_PATH, EMBEDDING_FILENAME))
low_dim_embedding = np.loadtxt(os.path.join(DATA_PATH, LOW_DIM_EMBEDDING_FILENAME))

with open(os.path.join(DATA_PATH, LABEL_FILENAME), encoding="utf-8") as f:
    lines = f.readlines()

labels = [label.strip() for label in lines]
label_vector = {}
for i, label in enumerate(labels):
    label_vector[label] = (low_dim_embedding[i, :], embedding[i, :])

LOW_DIM_EMBEDDING = 0
EMBEDDING = 1
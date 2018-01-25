from fastText import load_model
import os
from app import app

MODEL_PATH = app.config["MODEL_PATH"]
MODEL_FILENAME = app.config["MODEL_FILENAME"]

text_embedding_model = load_model(os.path.join(MODEL_PATH, MODEL_FILENAME))

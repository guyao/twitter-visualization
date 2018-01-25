from fastText import load_model

path = "model/"

text_embedding_model = load_model(path + "twitter.bin")

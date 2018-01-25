class Config(object):
    # APP
    APP_SECRET_KEY = "no more secret"

    # Data
    DATA_PATH = "data"
    EMBEDDING_FILENAME = "embedding"
    LOW_DIM_EMBEDDING_FILENAME = "low_dim_embedding"
    LABEL_FILENAME = "labels"

    # Model
    MODEL_PATH = "model"
    MODEL_FILENAME = "twitter.bin"

    # Visualization
    N_CHOICE = 30000 # number of random candidates used for compute distance
    N_NEIGHBOR = 400 # number of point on scatter

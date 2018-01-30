## Introduction
Unsupervised Hashtag Retrieval and Visualization for Crisis Informatics from Twitter

![](https://github.com/guyao/hashtag_twitter/raw/master/docs/images/screen_shot.png)

## Requirements
You will need

* [Python](https://www.python.org/) version 3
* [NumPy](http://www.numpy.org/) & [SciPy](https://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/)
* [fastText](https://fasttext.cc/)
* [mpld3](http://mpld3.github.io/)
* [flask](http://flask.pocoo.org/)

Note: Official release of `mpld3` does not support `numpy.ndarray`, which may cause compatible problems. You should install [this fork of repository](https://github.com/guyao/mpld3) instead.

## Structure
```
├── app
├── config.py
├── data
│   ├── README.md
│   ├── embedding
│   ├── labels
│   └── low_dim_embedding
├── docs
├── model
│   ├── twitter.bin
│   └── twitter.vec
├── requirements.txt
├── run.py
└── tools
    ├── low_dim_embedding.py
    ├── preprocessing.py
    ├── train.py
    └── vocab.py
```

## Installing dependencies
Besides `PyPI`, you also need to install

* Modified version of `mpld3`
```
pip install git+git://github.com/guyao/mpld3.git
```
* Python wrapper of `fastText` (pre-release)
```
pip install git+git://github.com/facebookresearch/fastText.git
```

It's recommended to use `pip` to install dependencies.

```
pip install -r requirements.txt
```



## Preprocessing twitter dump

`preprocessing.py`

```
usage: preprocessing.py [-h] [-z] [--hashtag] [--mentioned] input output

Twitter dump preprocessing tool

positional arguments:
  input          Input Filename
  output         Output Filename

optional arguments:
  -h, --help     show this help message and exit
  -z, --zipfile  input is zipfile
  --hashtag      Do not remove hashtags
  --mentioned    Do not remove mentioned
```

## Training text embedding model

`train.py`

```
usage: train.py [-h] input output

Model Training Tool

positional arguments:
  input       Input corpus filename
  output      Output model filename

optional arguments:
  -h, --help  show this help message and exit
```

## Get word vectors
To get words from a model, use:
```
$ python vocab.py model.bin words output.txt
```

To get word vectors, the program will read words from stdin. For example,
```
$ cat words.txt | python vocab.py model.bin vectors embedding.txt
```

`vocab.py`

```
usage: vocab.py [-h] model command output

Embedding word tool

positional arguments:
  model       Model to use
  command     command=[words|vectors]
  output      Output filename

optional arguments:
  -h, --help  show this help message and exit
```

## Get low dimension word vectors
Reduce dimension using [t-SNE](https://lvdmaaten.github.io/tsne/) for visualization.

```
$ python low_dim_embedding model.bin embedding.txt low_dim_embedding.txt
```

```
usage: low_dim_embedding.py [-h] [--multi] model input output

Embedding word tool

positional arguments:
  model       Model to use
  input       Input filename
  output      Output filename

optional arguments:
  -h, --help  show this help message and exit
  --multi     Use multi core tsne
```

Note: To accelerate this process in multicore machine, use [Multicore-TSNE](  https://github.com/DmitryUlyanov/Multicore-TSNE
)

## Launch the application
To configure the application, modify `config.py` as you need and place data/model in the data/model folder.

In the `data` folder, application requires labels, embedding vectors and low dimension vectors.
In the `model` folder, application requires the embedding model.

Launch the application use `run.py`, it will listen on port 5000 by default:
```
$ python run.py
```

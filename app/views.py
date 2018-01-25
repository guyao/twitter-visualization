from flask import render_template, request
import numpy as np
import sklearn
import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpld3
from app import app
from app.data import embedding, low_dim_embedding, labels, label_vector, LOW_DIM_EMBEDDING, EMBEDDING
from app.model import text_embedding_model

N_CHOICE = app.config["N_CHOICE"]
N_NEIGHBOR = app.config["N_NEIGHBOR"]


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


def process_query(q):
    q = q.strip()

    def is_hashtag(x): return x.startswith("#")

    if is_hashtag(q) and q in label_vector:
        v = label_vector[q]
    elif (not is_hashtag(q)) and ("#" + q) in label_vector:
        q = "#" + q
        v = label_vector[q]
    else:
        v = (None, process_nonexistent_word(q))
    return q, v


def process_nonexistent_word(w):
    return text_embedding_model.get_word_vector(w)


def sample_embedding():
    idx = np.arange(len(embedding))
    idx = np.random.choice(idx, N_CHOICE)

    embedding_sample = embedding[idx, :]
    low_dim_embedding_sample = low_dim_embedding[idx, :]
    label_sample = [labels[i] for i in idx]
    return embedding_sample, low_dim_embedding_sample, label_sample


@app.route('/api', methods=['POST', 'GET'])
def api():
    if request.method == "GET":
        q = request.args.get('q', '')
    elif request.method == "POST":
        q = request.form.get('q', '')

    q, vs = process_query(q)

    embedding_sample, low_dim_embedding_sample, label_sample = sample_embedding()

    inx_embedding = vs[EMBEDDING]
    inx_low_dim_embedding = vs[LOW_DIM_EMBEDDING]

    idx, _ = calc_n_cosine_neighbor(
        inx_embedding[np.newaxis, :], embedding_sample, N_NEIGHBOR)
    labels = [label_sample[i] for i in idx]

    # if query does not exist, it is hard to know it's low_dim_embedding coordinates
    # assume it locates in the cloest neighbor
    info = q
    if inx_low_dim_embedding is None:
        inx_low_dim_embedding = low_dim_embedding_sample[idx[-1], :]
        info = "Nonexistent word: " + q

    fig = plot_interactive_scatter(
        low_dim_embedding_sample[idx, :], labels, inx_low_dim_embedding[np.newaxis, :], q, info)

    return mpld3.fig_to_html(fig)


@app.route("/hashtags", methods=['GET'])
def hashtags():
    return render_template("hashtags.html", hashtags=label_vector.keys())


@app.route('/neighbor', methods=['POST', 'GET'])
@app.route('/neighbor/<q>')
def neighbor(q=None):
    if q is not None:
        pass
    else:
        if request.method == "GET":
            q = request.args.get('q', '')
        elif request.method == "POST":
            q = request.form.get('q', '')
    q = q.strip()

    q, vs = process_query(q)

    embedding_sample, low_dim_embedding_sample, label_sample = sample_embedding()

    inx_embedding = vs[EMBEDDING]

    idx, dist = calc_n_cosine_neighbor(
        inx_embedding[np.newaxis, :], embedding_sample, N_NEIGHBOR)
    labels = [label_sample[i] for i in idx]
    dists = [dist[i] for i in idx]
    neighbors = ["{} {}".format(t[0], str(t[1])) for t in zip(labels, dists)]

    return render_template("neighbor.html", hashtags=neighbors)


def calc_n_euclidean_neighbor(inX, X, N):
    distances = sklearn.metrics.pairwise.euclidean_distances(X, inX)
    sortedDist = distances.reshape((distances.shape[0],)).argsort()
    return sortedDist[:N]


def calc_n_cosine_neighbor(inX, X, N):
    distances = sklearn.metrics.pairwise.pairwise_distances(
        X, inX, metric="cosine")
    sortedDist = distances.reshape((distances.shape[0],)).argsort()
    return sortedDist[:N], distances


def plot_interactive_scatter(low_dim_embedding, labels, inx, q, info):
    from matplotlib.patches import Circle
    fig = plt.figure()
    plt.title(info)
    ax = fig.add_subplot(1, 1, 1)

    low_dim_embedding = np.concatenate([low_dim_embedding, inx])
    labels.append(q)

    # mark query
    c_x, c_y = inx[0]
    circle = Circle((c_x, c_y), 10, facecolor='none',
                    edgecolor='red', linewidth=3, alpha=0.5)
    ax.add_patch(circle)

    scatter = ax.scatter(
        low_dim_embedding[:, 0],
        low_dim_embedding[:, 1],
    )

    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    for i, label in enumerate(labels):
        x, y = low_dim_embedding[i, :]
        ax.text(x, y, label, alpha=0.4)
    mpld3.plugins.connect(fig, tooltip)
    return fig

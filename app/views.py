from flask import render_template, request
import numpy as np
import sklearn
import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpld3
from app import app
from app.data import X, labels, label_vector

N = 30000
N_neighbor = 400


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


@app.route('/api', methods=['POST', 'GET'])
def api():
    if request.method == "GET":
        q = request.args.get('q', '')
    elif request.method == "POST":
        q = request.form.get('q', '')
    q = q if q.startswith("#") else "#" + q
    q = q.strip()
    if q not in label_vector:
        return "Hashtag does not exist"

    idx = np.arange(len(X))
    idx = np.random.choice(idx, N)

    X_sample = X[idx, :]
    label_sample = [labels[i] for i in idx]

    inx = label_vector[q]
    idx = calc_n_neighbor(inx[np.newaxis, :], X_sample, N_neighbor)

    lbs = [label_sample[i] for i in idx]

    fig = plot_interactive_scatter(
        X_sample[idx, :], lbs, inx[np.newaxis, :], q)
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
    if not q:
        return "invalid input"
    q = q if q.startswith("#") else "#" + q

    if q not in label_vector:
        return "Hashtag does not exist"

    inx = label_vector[q]
    idx = calc_n_neighbor(inx[np.newaxis, :], X, N_neighbor)
    lbs = [labels[i] for i in idx]

    return render_template("neighbor.html", hashtags=lbs)


def calc_n_neighbor(inX, X, N):
    distances = sklearn.metrics.pairwise.euclidean_distances(X, inX)
    sortedDist = distances.reshape((distances.shape[0],)).argsort()
    return sortedDist[:N]


def plot_interactive_scatter(low_dim_embedding, labels, inx, q):
    from matplotlib.patches import Circle
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    low_dim_embedding = np.concatenate([low_dim_embedding, inx])
    labels.append(q)

    # mark query
    c_x, c_y = inx[0]
    circle = Circle((c_x, c_y), 1, facecolor='none',
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


@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        return "post"
    else:
        return "get"

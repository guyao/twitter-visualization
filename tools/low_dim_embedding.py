from fastText import load_model
import argparse
import numpy as np

# TSNE Config

N_COMPONENTS = 2  # should be 2 for 2D plot

# config
n_components = N_COMPONENTS
perplexity = 30.0
n_iter = 5000


def input(filename):
    return np.loadtxt(filename)


def output(low_dim_embedding, filename):
    np.savetxt(filename, low_dim_embedding)


def multi_core_tsne(embedding):
    # Muitlple core TSNE, much faster in specific environment
    # Require: MulticoreTSNE
    # https://github.com/DmitryUlyanov/Multicore-TSNE
    from MulticoreTSNE import MulticoreTSNE as TSNE
    import sklearn.metric
    tsne = TSNE(perplexity=perplexity, n_components=n_components,
                n_iter=n_iter, metric=sklearn.metrics.pairwise.cosine_distances)
    low_dim_embedding = tsne.fit_transform(embedding)
    return low_dim_embedding


def sklearn_tsne(embedding):
    from sklearn.manifold import TSNE
    tsne = TSNE(perplexity=perplexity, n_components=n_components,
                n_iter=n_iter, metric="cosine")
    low_dim_embedding = tsne.fit_transform(embedding)
    return low_dim_embedding


def process(input_filename, model, output_filename, tsne=sklearn_tsne):
    embedding = input(input_filename)
    low_dim_embedding = tsne(embedding)
    output(low_dim_embedding, output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Embedding word tool"
        )
    )

    parser.add_argument(
        "model",
        help="Model to use",
    )

    parser.add_argument(
        "input",
        help="Input filename",
    )

    parser.add_argument(
        "output",
        help="Output filename",
    )

    parser.add_argument(
        "--multi",
        help="Use multi core tsne",
        action='store_true',
        default=False,
    )

    args = parser.parse_args()
    model = load_model(args.model)
    tsne = multi_core_tsne if args.multi else sklearn_tsne
    process(args.input, model, args.output, tsne)

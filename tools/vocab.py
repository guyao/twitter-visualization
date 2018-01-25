from fastText import load_model
import argparse
import sys


def output(line, filename):
    with open(filename, "a") as f:
        f.write(line)
        f.write("\n")


def words(model, filename):
    for word in model.get_words():
        output(word, filename)


def vectors(model, filename):
    for word in sys.stdin:
        line = " ".join([str(i) for i in model.get_word_vector(word)])
        output(line, filename)


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
        "command",
        help="command=[words|vectors]",
    )

    parser.add_argument(
        "output",
        help="Output filename",
    )

    args = parser.parse_args()

    model = load_model(args.model)
    if args.command == "words":
        words(model, args.output)
    elif args.command == "vectors":
        vectors(model, args.output)

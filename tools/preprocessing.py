import json
import re
from collections import defaultdict
import argparse
import zipfile

trans_str = "!\"$%&\'()*+,-./:;<=>?[\\]^_`{|}~" + "…"
translate_table = str.maketrans(trans_str, " " * len(trans_str))

config = defaultdict(lambda: False)
config["hashtag"] = True
config["mentioned"] = True


def wash(j):
    s = j["text"]
    s = list(s.lower())
    st = 0
    if config["hashtag"]:
        if (j.get("entities") is not None) and j["entities"].get("hashtags") is not None:
            for h in j["entities"]["hashtags"]:
                l, r = h["indices"]
                s[l - st:r - st] = []
                st += r - l
    if s[0] == "r" and s[1] == "t":
        s[0:2] = []
    s = "".join(s)
    if config["mentioned"]:
        s = re.sub("@\w+", "", s)
    return " ".join(s.translate(translate_table).split())


def filter(j):
    if (j.get("lang") is not None) and (j["lang"] != "en"):
        return False
    return True


def output(output_filename, s):
    with open(output_filename, "a") as f:
        f.write(s)
        f.write("\n")

# Assume input is Twitter dump file, each line is a json object


def twitter_dump_process(input_filename, output_filename):
    global count
    with open(input_filename) as f:
        for line in f:
            j = json.loads(line)
            if filter(j):
                s = wash(j)
                output(output_filename, s)


def zip_process(input_filename, output_filename):
    zipf = zipfile.ZipFile(input_filename)
    lst = zipf.infolist()
    for l in lst[1:]:
        with zipf.open(l.filename) as f:
            j = json.load(f)
            if filter(j):
                s = wash(j)
                output(output_filename, s)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Twitter dump preprocessing tool"
        )
    )

    parser.add_argument(
        "-z",
        "--zipfile",
        help="input is zipfile",
        action='store_true',
        default=False,
    )

    parser.add_argument(
        "input",
        help="Input Filename",
    )
    parser.add_argument(
        "output",
        help="Output Filename",
    )

    parser.add_argument(
        "--hashtag",
        help="Do not remove hashtags",
        action='store_true',
        default=False,
    )

    parser.add_argument(
        "--mentioned",
        help="Do not remove mentioned",
        action='store_true',
        default=False,
    )

    args = parser.parse_args()
    if args.zipfile:
        process = zip_process
    else:
        processs = twitter_dump_process

    if args.hashtag:
        config["hashtag"] = False
    if args.mentioned:
        config["mentioned"] = False

    process(args.input, args.output)

from fastText import train_unsupervised
import argparse

# Config
model = "skipgram"
lr = 0.05
dim = 100
ws = 5
epoch = 5
minCount = 5
minCountLabel = 0
minn = 3
maxn = 6
neg = 5
wordNgrams = 1
loss = "ns"
bucket = 2000000
thread = 12
lrUpdateRate = 100
t = 1e-4
verbose = 2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Model Training Tool"
        )
    )

    parser.add_argument(
        "input",
        help="Input corpus filename",
    )
    parser.add_argument(
        "output",
        help="Output model filename",
    )

    args = parser.parse_args()

    input_filename = args.input
    output_filename = args.output

    model = train_unsupervised(
        input=input_filename,
        model=model,
        lr=lr,
        dim=dim,
        ws=ws,
        epoch=epoch,
        minCount=minCount,
        minCountLabel=minCountLabel,
        minn=minn,
        maxn=maxn,
        neg=neg,
        wordNgrams=wordNgrams,
        loss=loss,
        bucket=bucket,
        thread=thread,
        lrUpdateRate=lrUpdateRate,
        t=t,
        verbose=verbose,
    )

    model.save_model(output_filename)

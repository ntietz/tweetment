#!/bin/bash

if [ -e data/ ]; then
  echo "ERROR: data directory already exists."
  exit 1
fi

echo "Creating data directory..."
mkdir data/
mkdir data/raw/

curl https://www.cs.york.ac.uk/semeval-2013/task2/data/uploads/datasets/tweeti-b.dist.tsv > data/raw/training.tsv
curl https://www.cs.york.ac.uk/semeval-2013/task2/data/uploads/datasets/tweeti-b.dev.dist.tsv > data/raw/dev.tsv
curl https://www.cs.york.ac.uk/semeval-2013/task2/data/uploads/datasets/download_tweets.py > data/download_tweets.py

python data/download_tweets.py data/raw/training.tsv > data/training.tsv
python data/download_tweets.py data/raw/dev.tsv > data/dev.tsv


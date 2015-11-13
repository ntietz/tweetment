#!/bin/bash
# This script preprocesses a corpus by running the Twokenize software on it,
# yielding processable data.
# It takes in two arguments, the input path and output path.

INPUT=`pwd`/$1
OUTPUT=`pwd`/$2

if [ ! -d "./cache" ]; then
  echo 'Downloading the "Twokenize" software, this may take a while...'
  mkdir cache
  cd cache
  curl https://ark-tweet-nlp.googlecode.com/files/ark-tweet-nlp-0.3.2.tgz > twokenizer.tgz
  tar xfz twokenizer.tgz
  mv ark-tweet-nlp-0.3.2 twokenizer
  cd -
fi

if [ ! -e "./cache/clusters.csv" ]; then
  echo 'Downloading the Twitter word clusters provided by CMU researchers...'
  curl http://www.ark.cs.cmu.edu/TweetNLP/clusters/50mpaths2 > cache/clusters.csv
fi

cd cache/twokenizer
./runTagger.sh $INPUT > $OUTPUT


#!/bin/bash
# This script preprocesses a corpus by running the Twokenize software on it,
# yielding processable data.
# It takes in two arguments, the input path and output path.

if [ $# -lt 2 ]; then
  echo "Usage: ./classify.sh <model> <inputfile> <outputfile>"
  exit -1
fi

MODEL=$1
INPUT=$2
OUTPUT=$3

python main.py --classify --savefile $MODEL --input $INPUT --output $OUTPUT
rm $TMP


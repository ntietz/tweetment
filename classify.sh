#!/bin/bash
# This script preprocesses a corpus by running the Twokenize software on it,
# yielding processable data.
# It takes in two arguments, the input path and output path.

if [ $# -lt 2 ]; then
  echo "Usage: ./classify.sh <inputfile> <outputfile>"
  exit -1
fi

INPUT=`pwd`/$1
OUTPUT=$2
#TMP=/tmp/toktweets

#cd cache/twokenizer
#./runTagger.sh $INPUT > $TMP
#cd -

python main.py --classify --savefile ./cache/model.pkl --input $INPUT --output $OUTPUT
rm $TMP


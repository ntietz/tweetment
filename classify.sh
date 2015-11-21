#!/bin/bash
# This script preprocesses a corpus by running the Twokenize software on it,
# yielding processable data.
# It takes in two arguments, the input path and output path.

INPUT=`pwd`/$1
TMP=/tmp/toktweets

cd cache/twokenizer
./runTagger.sh $INPUT > $TMP
cd -

python main.py --classify --savefile ./cache/model.pkl --input $TMP
rm $TMP


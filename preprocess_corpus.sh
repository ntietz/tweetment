#!/bin/bash
# This script preprocesses a corpus by running the Twokenize software on it,
# yielding processable data.
# It takes in two arguments, the input path and output path.

INPUT=`pwd`/$1
OUTPUT=`pwd`/$2

cd cache/twokenizer
./runTagger.sh $INPUT > $OUTPUT


#!/bin/bash

cd input_data

../cache/twokenizer/twokenize.sh dev.gold.tsv > dev.tokens
../cache/twokenizer/twokenize.sh test.input.tsv > test.tokens
../cache/twokenizer/twokenize.sh training.gold.tsv > training.tokens

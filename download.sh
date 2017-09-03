#!/bin/bash

if [ ! -d "./cache" ]; then
  echo 'Downloading the "Twokenize" software, this may take a while...'
  mkdir cache
  cd cache
  curl https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/ark-tweet-nlp/ark-tweet-nlp-0.3.2.tgz > twokenizer.tgz
  tar xfz twokenizer.tgz
  mv ark-tweet-nlp-0.3.2 twokenizer
  cd -
fi

if [ ! -e "./cache/clusters.csv" ]; then
  echo 'Downloading the Twitter word clusters provided by CMU researchers...'
  curl http://www.cs.cmu.edu/~ark/TweetNLP/clusters/50mpaths2 > cache/clusters.csv
fi

curl http://saifmohammad.com/Lexicons/NRC-Emotion-Lexicon-v0.92.zip > cache/NRC-emotion-lexicon.zip
cd cache
unzip NRC-emotion-lexicon.zip
rm -rf __MACOSX/
cd ..

mkdir cache/bingliulexicon
curl https://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar > cache/bingliulexicon/lexicon.rar
cd cache/bingliulexicon
unrar e lexicon.rar
cd ../..

curl http://saifmohammad.com/WebDocs/NRC-Hashtag-Sentiment-Lexicon-v0.1.zip > cache/NRC-Hashtag-Sentiment-Lexicon-v0.1.zip
cd cache
unzip NRC-Hashtag-Sentiment-Lexicon-v0.1.zip
cd NRC-Hashtag-Sentiment-Lexicon-v0.1
gunzip bigrams-pmilexicon.txt.gz
gunzip unigrams-pmilexicon.txt.gz 
gunzip pairs-pmilexicon.txt.gz
cd ..
rm -rf __MACOSX
cd ..

curl http://saifmohammad.com/WebDocs/Sentiment140-Lexicon-v0.1.zip > cache/Sentiment140-Lexicon-v0.1.zip
cd cache
unzip Sentiment140-Lexicon-v0.1.zip
cd Sentiment140-Lexicon-v0.1/
gunzip pairs-pmilexicon.txt.gz
gunzip bigrams-pmilexicon.txt.gz
gunzip unigrams-pmilexicon.txt.gz
cd ..
rm -rf __MACOSX
cd ..

echo "WARNING: Cannot automatically download the MPQA Subjectivity Lexicon."
echo " Please download it manually and place it in the cache directory with the others."
echo " http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/"
echo " You should end up with a directory: ./cache/subjectivity_clues_hltemnlp05."

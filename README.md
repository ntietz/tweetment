Overview
========

This library contains a pre-trained sentiment classifier for Twitter data. It
is an implementation of work done by Mohammad et al (information below).
An overview of usage is provided below. An overview of the technical details
can be found in the paper, referenced below.

Note that this is not an example of superb engineering; it is just research
code to serve a purpose, and could definitely be improved in terms of
organization and efficiency, but it does the job for now.

Right now, the library achieves an F1 score of 62.41 on the SemEval-2013 B
dataset. The paper's implementation achieved an F1 score of 69.02. Obviously,
this is better performance. I have omitted two features: parts of speech and
negation. I will add negation in the future. Parts of speech was omitted due to
lack of a Python library for doing this. Again, getting this to work is future
work.

**Contact**: Nicole Tietz-Sokolskaya (me@ntietz.com)

Basic Usage
===========

Requirements
------------

Requires **scikit-learn** version 0.17. Not tested with any others.

You must download the model from the latest release. Download the file
**model.pkl** and place it wherever you'd like.

Go over to the [release page](https://github.com/ntietz/tweetment/releases) to
get the latest.

Usage
-----

To classify tweets, you just need a file with one message per line:

```
<tweet1>
<tweet2>
...
```

Then you simply run classify.sh over this file:

```
./classify.sh model.pkl tweets.txt results.txt
```

The results are printed out into the file specified. The output format is as
follows (but tab separated):

```
<sentiment1> <tweet1>
<sentiment2> <tweet2>
...
```

You will notice that this takes some time. This is mostly the startup time to
load the saved model off the disk. The actual classification of tweets is not
too expensive.

That's it!

Using as a Library
==================

Usage as a library is really simple. Basically, you just construct a classifier
and tell it where to load the model from, then you can directly classify
tweets without preprocessing them.

Here's a working example of how to do this:

```
import tweetment
import datetime

print "%s: Loading model." % (datetime.datetime.now().time())
classifier = tweetment.SentimentClassifier('./cache/model.pkl')
print "%s: Model is loaded." % (datetime.datetime.now().time())

for line in open('example_tweets.txt'):
  tweet = line.strip()
  sentiment = classifier.classify(tweet)
  print "%s: Classified tweet as %s." % (datetime.datetime.now().time(), sentiment)
```

Following is the expected output, using my 10 sample tweets. The times may vary
on your machine, but are provided here to give you an idea of the speed of
loading vs the speed of classification.

```
17:39:05.610987: Loading model.
17:39:27.308373: Model is loaded.
17:39:27.342119: Classified tweet as positive.
17:39:27.371931: Classified tweet as neutral.
17:39:27.397071: Classified tweet as positive.
17:39:27.427360: Classified tweet as negative.
17:39:27.453307: Classified tweet as negative.
17:39:27.480550: Classified tweet as neutral.
17:39:27.507628: Classified tweet as positive.
17:39:27.534367: Classified tweet as neutral.
17:39:27.561039: Classified tweet as positive.
17:39:27.590164: Classified tweet as neutral.
```

Training It Yourself
====================

First Steps
-----------

**Important: you must do this first, or the library will break.**
You have to do this even if you are not training it yourself.

You must download some lexicons. Most of these are done automatically by running
my script, but there is one you must download yourself (instructions will be printed at the end). First run the script:

```
./download.sh
```

This will download and prepare the lexicons for you, and will also print some
directions for the other lexicon. If this doesn't work, email me and I'll help.

Your input data is expected in the following format (tab-separated values):
```
<num> <num> <label> <tweet>
```
The first two numbers are tweet and author identifiers, and are not used, but are artifacts from the original dataset this library was based off of.

*label* is a classification ("positive", "negative", or "neutral") of the tweet.

*tweet* is the content of the tweet you are classifying.


Now, the data must be split into the training, dev, and test portions. Take your input file (`input.tsv`) and split it into three files (apportioned how you would like): `training.gold.tsv`, `dev.gold.tsv`, and `test.input.tsv`. These should all be in the `input_data` directory.

This data must be tokenized (or "twokenized"), so run the twokenizer on each like so:
```
./prep_input.sh
```

Now, we just have to do the training step.
```
python main.py --train --input input_data/ --clusters cache/clusters.csv --cache cache/
```

Licensing and Attribution
=========================

TweetMotif
----------

TweetMotif is licensed by Brendan O'Connor. The latest version can be found on
GitHub (https://github.com/brendano/tweetmotif). This software is distributed
with mine and is licensed under the Apache license. The license for this
software is included in the ./tweetmotif directory, as well as copyright
information.

TweetMotif is licensed under the Apache License 2.0: http://www.apache.org/licenses/LICENSE-2.0.html

Copyright Brendan O'Connor, Michel Krieger, and David Ahn, 2009-2010

Tweetment
---------

The Sentiment library was developed by Nicole Tietz-Sokolskaya. It is released under the
GPL; see the LICENSE file.

Copyright 2015-2017 Nicole Tietz-Sokolskaya

This library is an implementation of work done by Saif M. Mohammad, Svetlana
Kiritchenko, and Xiaodan Zhu. Please see their paper "NRC-Canada: Building the
State-of-the-Art in Sentiment Analysis of Tweets" for more details. It is
here: http://saifmohammad.com/WebDocs/sentimentMKZ.pdf.


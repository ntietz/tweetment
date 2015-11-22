Overview
========

This library contains a pre-trained sentiment classifier for Twitter data. It
is an implementation of work done by Mohammad et al (information below).
An overview of usage is provided below.

**Contact**: Nicole Tietz-Sokolskaya (ntietz@gmail.com)

Basic Usage
===========

Requirements
------------

Requires **scikit-learn** version 0.17. Not tested with any others.

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
./classify.sh tweets.txt results.txt
```

TODO

Using as a Library
==================

TODO

Training It Yourself
====================

First Steps
-----------

**Important: you must do this first, or the library will break.**
You have to do this even if you are not training it yourself.

You must download some lexicons. Most of these are done automatically by running
my script, but there is one you must download yourself. First run the script:

```
cd scripts
./download.sh
```

This will download and prepare the lexicons for you, and will also print some
directions for the other lexicon. If this doesn't work, email me and I'll help.

TODO


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

Sentiment
---------

The Sentiment library was developed by Nicole Tietz-Sokolskaya. I want to license it
under the Apache License 2.0: http://www.apache.org/licenses/LICENSE-2.0.html
For now, do not assume that that is true. I am waiting to find out the policy
of our university on software licenses.

Copyright 2015 (It is currently unclear to me whether I hold the copyright or the university does, so I am leaving that out for now).

This library is an implementation of work done by Saif M. Mohammad, Svetlana
Kiritchenko, and Xiaodan Zhu. Please see their paper "NRC-Canada: Building the
State-of-the-Art in Sentiment Analysis of Tweets" for more details. It is
here: http://saifmohammad.com/WebDocs/sentimentMKZ.pdf.


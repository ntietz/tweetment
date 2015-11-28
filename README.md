Overview
========

This library contains a pre-trained sentiment classifier for Twitter data. It
is an implementation of work done by Mohammad et al (information below).
An overview of usage is provided below. An overview of the technical details
can be found in the paper, referenced below.

Note that this is not an example of superb engineering; it is just research
code to serve a purpose, and could definitely be improved in terms of
organization and efficiency, but it does the job for now.

**Contact**: Nicole Tietz-Sokolskaya (me@ntietz.com)

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

**TODO**

Instructions for this and code samples are coming soon! If you want to use this
and instructions aren't here yet, **please** email me or open an issue and I
will complete this ASAP.

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

**TODO**

Instructions for this and code samples are coming soon! If you want to use this
and instructions aren't here yet, **please** email me or open an issue and I
will complete this ASAP.

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
under the GPL but I do not know if I am permitted by the university's policy.
For now, do not assume that that is true. I am waiting to find out the policy
of our university on software licenses.

Copyright 2015 (Either Nicole or The Ohio State University; I am waiting to hear what the university's policy is)

This library is an implementation of work done by Saif M. Mohammad, Svetlana
Kiritchenko, and Xiaodan Zhu. Please see their paper "NRC-Canada: Building the
State-of-the-Art in Sentiment Analysis of Tweets" for more details. It is
here: http://saifmohammad.com/WebDocs/sentimentMKZ.pdf.


import argparse
import itertools
from sklearn import svm
import string
import tweetmotif.twokenize
import tweetmotif.emoticons as emoticons
import pickle
import sys

class SentimentClassifier:
  '''
    This class contains a sentiment classifier for tweets.
  '''
  def __init__(self, saved_model=None):

    self.w_ngram_counts = {}
    self.c_ngram_counts = {}
    self.n_ngram_counts = {}

    if saved_model is not None:
      with open(saved_model, 'rb') as savefile:
        self.model = pickle.load(savefile)
        self.classifier = self.model['classifier']


  def classify_file(self, in_name, out_name):
    features = []
    tweets = []

    with open(in_name) as f:
      for line in f:
        if len(line.strip()) == 0:
          sys.stderr.write('WARNING: blank line in input detected. Skipping it.\n')
          continue
        tweet = line.split('\t')[0]
        features.append(self.generate_features(tweet, self.model['w2c'], self.model['cids'], self.model['word_ngrams'], self.model['nonc_ngrams'], self.model['char_ngrams'], self.model['lexicons']))
        tweets.append(tweet)

    predictions = self.classifier.predict(features)
    with open(out_name, 'w') as f:
      for p, tweet in zip(predictions, tweets):
        label = self.model['int_to_label'][p]
        f.write('%s\t%s' % (label, tweet))


  def classify(self, tweet):
    features = self.generate_features(tweet, self.model['w2c'], self.model['cids'], self.model['word_ngrams'], self.model['nonc_ngrams'], self.model['char_ngrams'], self.model['lexicons'])
    predictions = self.classifier.predict([features])
    return self.model['int_to_label'][predictions[0]]


  def generate_features(self, tweet, w2c, cids, corpus_word_ng,
      corpus_nonc_ng, corpus_char_ng, lexicons):
    '''
      Takes in a tweet and generates a feature vector.
    '''
    words = self._tokenize(tweet)

    '''
      Our features are: (+ done, / wip, - todo)
      + allcaps
      - pos
      + hashtags
      + punctuation
      + elongated words
      + clusters
      + word ngrams
      + character ngrams
      + emoticons
      + lexicons
      - negation
    '''

    num_allcaps = 0
    num_hashtags = 0
    num_elongated = 0
    for word in words:
      if self._word_is_all_caps(word):
        num_allcaps += 1
      if word[0] == '#':
        num_hashtags += 1
      if self._is_elongated(word):
        num_elongated += 1

    last_is_question_or_exclaim = 1 if self._contains_question_or_exclaim(words[-1]) else 0
    num_seq_question, num_seq_exclaim, num_seq_both = self._num_contiguous_question_exclaim(tweet)
    cluster_mem_vec = self._get_cluster_mem_vec(cids, w2c, words)

    ngram_w_vec, ngram_n_vec, ngram_c_vec = self._get_ngram_vec(tweet, corpus_word_ng,
        corpus_nonc_ng, corpus_char_ng)

    emoticon_vec = self._get_emoticon_vec(tweet, words[-1])

    lexicon_vec = self._get_lexicon_vec(words, lexicons)

    features = [num_allcaps, num_hashtags, num_elongated,
        last_is_question_or_exclaim, num_seq_question, num_seq_exclaim,
        num_seq_both] + cluster_mem_vec + ngram_w_vec + ngram_n_vec +\
        ngram_c_vec + emoticon_vec + lexicon_vec
    return features

  def _contains_question_or_exclaim(self, token):
    for c in token:
      if c == '?' or c == '!':
        return True
    return False

  def _num_contiguous_question_exclaim(self, tweet):
    num_q = 0
    num_e = 0
    num_qe = 0
    entered_seq = False
    mixed = False
    last_was = ''
    for c in tweet + ' ': # add an extra space for an extra iteration (hack)
      if entered_seq:
        if c != '!' and c != '?':
          # End of sequence
          if mixed:
            num_qe += 1
          elif last_was == '!':
            num_e += 1
          else:
            num_q += 1
          entered_seq = False
          mixed = False
        elif c != last_was:
          # Mixture of ! and ? must be here
          mixed = True
      else:
        if c == '!' or c == '?':
          entered_seq = True
      last_was = c
    return num_q, num_e, num_qe

  def _get_cluster_mem_vec(self, cids, w2c, words):
    vec = [0]*len(cids)
    for word in words:
      if word in w2c:
        vec[cids[w2c[word]]] = 1
    return vec


  def _get_ngram_vec(self, tweet, corpus_word_ng, corpus_nonc_ng, corpus_char_ng):
    word_vec = [0] * len(corpus_word_ng)
    nonc_vec = [0] * len(corpus_nonc_ng)
    char_vec = [0] * len(corpus_char_ng)

    word_lengths = [1,2,3,4]
    char_lengths = [1,2,3]

    for length in char_lengths:
      for idx in range(0, len(tweet) - length + 1):
        ng = tweet[idx : idx+length]
        if ng in corpus_char_ng:
          char_vec[corpus_char_ng[ng]] = 1

    words = self._tokenize(tweet)
    for length in word_lengths:
      for idx in range(0, len(words) - length + 1):
        ng = words[idx : idx+length]
        if tuple(ng) in corpus_word_ng:
          word_vec[corpus_word_ng[tuple(ng)]] = 1
        for j in range(0, length):
          tmp = list(ng)
          tmp[j] = '*' # TODO: should this be a symbol like <*> ?
          if tuple(tmp) in corpus_nonc_ng:
            nonc_vec[corpus_nonc_ng[tuple(tmp)]] = 1

    return word_vec, nonc_vec, char_vec


  def _get_emoticon_vec(self, original_tweet, last_token):
    '''
      Checks if the passed-in string contains any emoticons.

      Uses TweetMotif.
    '''
    positive_in_tweet = 1 if emoticons.Happy_RE.search(original_tweet) else 0
    negative_in_tweet = 1 if emoticons.Sad_RE.search(original_tweet) else 0
    positive_ends_tweet = 1 if emoticons.Happy_RE.search(last_token) else 0
    negative_ends_tweet = 1 if emoticons.Sad_RE.search(last_token) else 0
    return [positive_in_tweet, negative_in_tweet, positive_ends_tweet,
        negative_ends_tweet]


  def _get_lexicon_vec(self, tokens, lexicons):
    '''
      Takes in the tweet (tokenized) and the lexicons and returns a feature
      vector for the tweet.
    '''
    vec = []
    lexicon_names = ['bingliu', 'nrc-emotion', 'mpqa', 'hashtag', 'sentiment140']

    # Unigrams!
    for sentiment in ['positive', 'negative', 'neutral']:
      scores = {}
      for lexicon in lexicon_names:
        scores[lexicon] = {}
        scores[lexicon]['count'] = 0
        scores[lexicon]['total'] = 0.0
        scores[lexicon]['max'] = 0
        scores[lexicon]['last'] = 0

      for token in tokens:
        for lexicon in lexicon_names:
          if token in lexicons[lexicon][sentiment]:
            scores[lexicon]['count'] += 1
            scores[lexicon]['total']  += lexicons[lexicon][sentiment][token]
            scores[lexicon]['max']  = max(lexicons[lexicon][sentiment][token], scores[lexicon]['max'])
            scores[lexicon]['last'] = lexicons[lexicon][sentiment][token]

      for lexicon in lexicon_names:
        vec += [scores[lexicon]['count'], scores[lexicon]['total'], scores[lexicon]['max'], scores[lexicon]['last']]

    # For bigrams and pairs, we only use a subset.
    lexicon_names = ['hashtag', 'sentiment140']

    # Bigrams!
    for sentiment in ['positive', 'negative', 'neutral']:
      scores = {}
      for lexicon in lexicon_names:
        scores[lexicon] = {}
        scores[lexicon]['count'] = 0
        scores[lexicon]['total'] = 0.0
        scores[lexicon]['max'] = 0
        scores[lexicon]['last'] = 0

      for bigram in zip(tokens, tokens[1:]):
        token = bigram[0] + ' ' + bigram[1]
        for lexicon in lexicon_names:
          if token in lexicons[lexicon][sentiment]:
            scores[lexicon]['count'] += 1
            scores[lexicon]['total']  += lexicons[lexicon][sentiment][token]
            scores[lexicon]['max']  = max(lexicons[lexicon][sentiment][token], scores[lexicon]['max'])
            scores[lexicon]['last'] = lexicons[lexicon][sentiment][token]

      for lexicon in lexicon_names:
        vec += [scores[lexicon]['count'], scores[lexicon]['total'], scores[lexicon]['max'], scores[lexicon]['last']]

    # Pairs!
    for sentiment in ['positive', 'negative', 'neutral']:
      scores = {}
      for lexicon in lexicon_names:
        scores[lexicon] = {}
        scores[lexicon]['count'] = 0
        scores[lexicon]['total'] = 0.0
        scores[lexicon]['max'] = 0
        scores[lexicon]['last'] = 0

      for pair in itertools.permutations(tokens, 2):
        token = pair[0] + '---' + pair[1]
        for lexicon in lexicon_names:
          if token in lexicons[lexicon][sentiment]:
            scores[lexicon]['count'] += 1
            scores[lexicon]['total']  += lexicons[lexicon][sentiment][token]
            scores[lexicon]['max']  = max(lexicons[lexicon][sentiment][token], scores[lexicon]['max'])
            scores[lexicon]['last'] = lexicons[lexicon][sentiment][token]

      for lexicon in lexicon_names:
        vec += [scores[lexicon]['count'], scores[lexicon]['total'], scores[lexicon]['max'], scores[lexicon]['last']]

    return vec


  def _tokenize(self, tweet):
    t = tweetmotif.twokenize.tokenize(tweet)
    return t


  def _corpus_ngrams(self, corpus):
    '''
      Assumes that corpus is a list of tweets, and that each tweet is a space
      separated list of tokens (we can split on spaces).
    '''
    word_lengths = [1,2,3,4]
    char_lengths = [1,2,3]

    word_ngrams = {}
    nonc_ngrams = {} # non-contiguous word ngrams
    char_ngrams = {}

    word_idx = 0
    nonc_idx = 0
    char_idx = 0

    for record in corpus:
      tweet = record[0]
      for length in char_lengths:
        for idx in range(0, len(tweet) - length + 1):
          ng = tweet[idx : idx+length]
          if ng not in char_ngrams:
            char_ngrams[ng] = char_idx
            char_idx += 1
            self.c_ngram_counts[ng] = 1
          else:
            self.c_ngram_counts[ng] += 1
      words = self._tokenize(tweet)
      for length in word_lengths:
        for idx in range(0, len(words) - length + 1):
          ng = words[idx : idx+length]
          if tuple(ng) not in word_ngrams:
            word_ngrams[tuple(ng)] = word_idx
            word_idx += 1
            self.w_ngram_counts[tuple(ng)] = 1
          else:
            self.w_ngram_counts[tuple(ng)] += 1
          for j in range(0, length):
            tmp = list(ng)
            tmp[j] = '*' # TODO: should this be a symbol like <*> ?
            if tuple(tmp) not in nonc_ngrams:
              nonc_ngrams[tuple(tmp)] = nonc_idx
              nonc_idx += 1
              self.n_ngram_counts[tuple(tmp)] = 1
            else:
              self.n_ngram_counts[tuple(tmp)] += 1

    smaller_word_ngrams = {}
    word_idx = 0
    for key in word_ngrams.keys():
      if self.w_ngram_counts[key] > 1:
        smaller_word_ngrams[key] = word_idx
        word_idx += 1

    smaller_nonc_ngrams = {}
    nonc_idx = 0
    for key in nonc_ngrams.keys():
      if self.n_ngram_counts[key] > 1:
        smaller_nonc_ngrams[key] = nonc_idx
        nonc_idx += 1

    smaller_char_ngrams = {}
    char_idx = 0
    for key in char_ngrams.keys():
      if self.c_ngram_counts[key] > 1:
        smaller_char_ngrams[key] = char_idx
        char_idx += 1

    print "Before removing single occurrence ngrams, we had %s ngrams." % (len(word_ngrams) + len(nonc_ngrams) + len(char_ngrams))
    print "After reduction, we have %s ngrams." % (len(smaller_word_ngrams) + len(smaller_nonc_ngrams) + len(smaller_char_ngrams))

    return smaller_word_ngrams, smaller_nonc_ngrams, smaller_char_ngrams


  def _word_is_all_caps(self, word):
    for c in word:
      if c not in string.ascii_uppercase:
        return False
    return True


  def _is_elongated(self, word):
    elong_len = 3
    for idx in range(len(word) - elong_len + 1):
      if word[idx] == word[idx+1] and word[idx] == word[idx+2]:
        return True
    return False


  def _load_clusters(self, cluster_filename):
    clusters_to_words = {}
    words_to_clusters = {}
    clusters = {}
    idx = 0
    with open(cluster_filename) as f:
      for line in f:
        (cluster, word, count) = line.split()
        if not cluster in clusters_to_words:
          clusters_to_words[cluster] = set()
          clusters[cluster] = idx
          idx += 1
        clusters_to_words[cluster].add(word)
        words_to_clusters[word] = cluster
    return words_to_clusters, clusters_to_words, clusters


  def _load_lexicons(self, cache_dir):
    '''
      Pass in the path to the cache directory, and this function will load return
      the lexicons we are using.

      Our standard lexicon format will be:
        {
          'positive': dict(...),
          'negative': dict(...),
          'neutral': dict(...)
        }
      A term has the sentiment of whichever dictionary it is in (possibly
      multiple).
    '''
    lexicons = {}
    # First, load the Bing Liu lexicon.
    # NOTE: the paper we are replicating does not indicate what scores are used
    # for this lexicon, so we used 1.0 for each term.
    lexicons['bingliu'] = {'positive': {}, 'negative': {}, 'neutral': {}}
    with open(cache_dir + '/bingliulexicon/positive-words.txt') as pos_file:
      for line in pos_file:
        if line[0] == ';' or len(line) == 0:
          continue # skip comments or blank lines
        lexicons['bingliu']['positive'][line.strip()] = 1.0
    with open(cache_dir + '/bingliulexicon/negative-words.txt') as neg_file:
      for line in neg_file:
        if line[0] == ';' or len(line) == 0:
          continue # skip comments or blank lines
        lexicons['bingliu']['negative'][line.strip()] = 1.0

    # Now load the NRC Emotion Lexicon.
    # NOTE: the paper we are replicating does not indicate what scores are used
    # for this lexicon, so we used 1.0 for each term.
    lexicons['nrc-emotion'] = {'positive': {}, 'negative': {}, 'neutral': {}}
    with open(cache_dir + '/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt') as f:
      line_number = 0
      for line in f:
        if line_number < 46:
          line_number += 1
          continue
        word, affect, flag = line.strip().split()
        if flag == '1':
          if affect == 'positive':
            lexicons['nrc-emotion']['positive'][word] = 1.0
          elif affect == 'negative':
            lexicons['nrc-emotion']['negative'][word] = 1.0

    # Now load the MPQA lexicon.
    # NOTE: the paper we are replicating does not indicate what scores are used
    # for this lexicon, so we used 1.0 for each term.
    # NOTE: this lexicon is deeper than we are using it for. It also gives info
    # about stemming, parts of speech, etc. which we ignore for now.
    lexicons['mpqa'] = {'positive': {}, 'negative': {}, 'neutral': {}}
    with open(cache_dir + '/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff') as f:
      for line in f:
        word_idx = line.find('word')
        word = line[ line.find('=',word_idx)+1 : line.find(' ',word_idx) ]

        # NOTE: some of the polarities here are weird, so I'm just going to punt
        # and only use 'positive', 'negative', or 'neutral'. I'm ignoring weak
        # polarities and 'both'.
        positive_idx = line.rfind('positive')
        negative_idx = line.rfind('negative')
        neutral_idx = line.rfind('neutral')
        # 'positive', 'negative', and 'neutral' only co-occur once in the entire
        # lexicon, and in that case the final occurrence is the correct sentiment
        # so we will apply that technique to the whole dataset.
        if positive_idx > max(negative_idx, neutral_idx):
          lexicons['mpqa']['positive'][word] = 1.0
        elif negative_idx > max(positive_idx, neutral_idx):
          lexicons['mpqa']['negative'][word] = 1.0
        else:
          lexicons['mpqa']['neutral'][word] = 1.0

    lexicons['hashtag'] = {'positive': {}, 'negative': {}, 'neutral': {}}
    with open(cache_dir + '/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt') as f:
      for line in f:
        word, score_str, _, _ = line.split()
        score = float(score_str)
        if score > 0:
          lexicons['hashtag']['positive'][word] = score
        else:
          lexicons['hashtag']['negative'][word] = abs(score)
    with open(cache_dir + '/NRC-Hashtag-Sentiment-Lexicon-v0.1/bigrams-pmilexicon.txt') as f:
      for line in f:
        word, score_str, _, _ = line.split('\t')
        score = float(score_str)
        if score > 0:
          lexicons['hashtag']['positive'][word] = score
        else:
          lexicons['hashtag']['negative'][word] = abs(score)
    with open(cache_dir + '/NRC-Hashtag-Sentiment-Lexicon-v0.1/pairs-pmilexicon.txt') as f:
      for line in f:
        word, score_str, _, _ = line.split('\t')
        score = float(score_str)
        if score > 0:
          lexicons['hashtag']['positive'][word] = score
        else:
          lexicons['hashtag']['negative'][word] = abs(score)

    lexicons['sentiment140'] = {'positive': {}, 'negative': {}, 'neutral': {}}
    with open(cache_dir + '/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt') as f:
      for line in f:
        word, score_str, _, _ = line.split()
        score = float(score_str)
        if score > 0:
          lexicons['sentiment140']['positive'][word] = score
        else:
          lexicons['sentiment140']['negative'][word] = abs(score)
    with open(cache_dir + '/Sentiment140-Lexicon-v0.1/bigrams-pmilexicon.txt') as f:
      for line in f:
        word, score_str, _, _ = line.split('\t')
        score = float(score_str)
        if score > 0:
          lexicons['sentiment140']['positive'][word] = score
        else:
          lexicons['sentiment140']['negative'][word] = abs(score)
    with open(cache_dir + '/Sentiment140-Lexicon-v0.1/pairs-pmilexicon.txt') as f:
      for line in f:
        word, score_str, _, _ = line.split('\t')
        score = float(score_str)
        if score > 0:
          lexicons['sentiment140']['positive'][word] = score
        else:
          lexicons['sentiment140']['negative'][word] = abs(score)

    return lexicons


  def train(self, inputdir, cache, clusters, modelout):
    # First, we want to train the classifier
    training_gold = open(inputdir + '/training.gold.tsv')
    training_tokens = open(inputdir + '/training.tokens')
    dev_gold = open(inputdir + '/dev.gold.tsv')
    dev_tokens = open(inputdir + '/dev.tokens')
    test_input = open(inputdir + '/test.input.tsv')
    test_tokens = open(inputdir + '/test.tokens')

    gold_lines = [line.strip() for line in training_gold]
    token_lines = [line.strip() for line in training_tokens]
    gold_lines += [line.strip() for line in dev_gold]
    token_lines += [line.strip() for line in dev_tokens]
    test_input_lines = [line for line in test_input]
    test_token_lines = [line.strip() for line in test_tokens]
    assert(len(gold_lines) == len(token_lines))
    print "Loaded %s training examples." % len(gold_lines)

    label_to_int = {'"positive"': 0, '"neutral"': 1, '"objective-OR-neutral"': 1, '"negative"': 2}
    int_to_label = {0: 'positive', 1: 'neutral', 2: 'negative'}

    training_positive = []
    training_negative = []
    training_neutral = []

    training_corpus = map(lambda x: x.split('\t'), token_lines)
    word_ngrams, nonc_ngrams, char_ngrams = self._corpus_ngrams(training_corpus)
    print "Generated ngram encodings for training corpus."

    print "Contains %s @ mentions." % len(filter(lambda x: len(x) == 1 and x[0][0] == '@', word_ngrams.keys()))
    #print "Contains %s used only once." % len(filter(lambda x: ngram_counts[x] == 1, word_ngrams.keys()))
    print "Contains %s URLs." % len(filter(lambda x: len(x) == 1 and x[0][:4] == 'http', word_ngrams.keys()))

    lexicons = self._load_lexicons(cache)
    print "Loaded the lexicons."

    w2c, c2w, cids = self._load_clusters(clusters)
    print "Loaded the clusters."

    training_features = []
    training_classes = []

    for gold_line, tokenized_line in zip(gold_lines, token_lines):
      _, _, label, _ = gold_line.split('\t')
      tweet = tokenized_line.split('\t')[0]

      features = self.generate_features(tweet, w2c, cids, word_ngrams, nonc_ngrams, char_ngrams, lexicons)
      training_features.append(features)
      training_classes.append(label_to_int[label])

      if len(training_features) % 1000 == 0:
        print "Loaded %s feature vectors." % len(training_features)

    test_features = []
    for tokenized_line in test_token_lines:
      tweet = tokenized_line.split('\t')[3]
      features = self.generate_features(tweet, w2c, cids, word_ngrams, nonc_ngrams, char_ngrams, lexicons)
      test_features.append(features)

    classifier = svm.LinearSVC(C=0.005)
    print "Created classifier. Training..."
    classifier.fit(training_features, training_classes)
    print "Trained classifier."
    print "Predicting %s test cases." % len(test_features)
    test_predictions = classifier.predict(test_features)
    print "Finished prediction. Outputting now."

    with open('test_predictions.txt', 'w') as fout:
      for (prediction, line) in zip(test_predictions, test_input_lines):
        col1, col2, _, tweet = line.split('\t')
        label = int_to_label[prediction]
        fout.write('%s\t%s\t%s\t%s' % (col1, col2, label, tweet))
    print "Done outputting predictions."

    print "Saving model..."
    with open(modelout, 'wb') as savefile:
      model = {
          'label_to_int': label_to_int,
          'int_to_label': int_to_label,
          'word_ngrams': word_ngrams,
          'nonc_ngrams': nonc_ngrams,
          'char_ngrams': char_ngrams,
          'lexicons': lexicons,
          'w2c': w2c,
          'c2w': c2w,
          'cids': cids,
          'classifier': classifier
          }
      pickle.dump(model, savefile)



import argparse
import sklearn
import string
import tweetmotif.emoticons as emoticons

# This list of POS tags is provided in the annotation guidelines for Twokenizer.
# See here: http://www.ark.cs.cmu.edu/TweetNLP/annot_guidelines.pdf
pos_tags = ['N', 'O', '^', 'S', 'Z', 'V', 'A', 'R', '!', 'D', 'P', '&', 'T', 'X', '#', '@', '~', 'U', 'E', '$', ',', 'G', 'L', 'M', 'Y']
pos_idxs = dict(zip(pos_tags, range(0, len(pos_tags))))


def corpus_ngrams(corpus):
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
    tweet, _, _, _ = record
    for length in char_lengths:
      for idx in range(0, len(tweet) - length + 1):
        ng = tweet[idx : idx+length]
        if ng not in char_ngrams:
          char_ngrams[ng] = char_idx
          char_idx += 1
    words = tweet.split()
    for length in word_lengths:
      for idx in range(0, len(words) - length + 1):
        ng = words[idx : idx+length]
        if tuple(ng) not in word_ngrams:
          word_ngrams[tuple(ng)] = word_idx
          word_idx += 1
        for j in range(0, length):
          tmp = list(ng)
          tmp[j] = '*' # TODO: should this be a symbol like <*> ?
          if tuple(tmp) not in nonc_ngrams:
            nonc_ngrams[tuple(tmp)] = nonc_idx
            nonc_idx += 1

  #print "words: %s \n\n nonc: %s \n\n char: %s \n\n" % (word_ngrams, nonc_ngrams, char_ngrams)
  return word_ngrams, nonc_ngrams, char_ngrams


def get_ngram_vec(tweet, corpus_word_ng, corpus_nonc_ng, corpus_char_ng):
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

  words = tweet.split()
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


def word_is_all_caps(word):
  for c in word:
    if c not in string.ascii_uppercase:
      return False
  return True


def contains_question_or_exclaim(token):
  for c in token:
    if c == '?' or c == '!':
      return True
  return False


def num_contiguous_question_exclaim(tweet):
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


def is_elongated(word):
  elong_len = 3
  for idx in range(len(word) - elong_len + 1):
    if word[idx] == word[idx+1] and word[idx] == word[idx+2]:
      return True
  return False


def get_emoticon_vec(original_tweet, last_token):
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


def load_clusters(cluster_filename):
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


def get_cluster_mem_vec(cids, w2c, words):
  vec = [0]*len(cids)
  for word in words:
    if word in w2c:
      vec[cids[w2c[word]]] = 1
  return vec


def get_pos_vec(pos):
  vec = [0]*len(pos_tags)
  for tag in pos:
    vec[pos_idxs[tag]] += 1
  return vec


def generate_features(record, w2c, cids, corpus_word_ng,
    corpus_nonc_ng, corpus_char_ng):
  '''
    Takes in a tweet and generates a feature vector.
  '''
  tweet, pos, pos_conf, orig = record
  words = tweet.split()

  '''
    Our features are: (+ done, / wip, - todo)
    + allcaps
    + pos
    + hashtags
    + punctuation
    + elongated words
    + clusters
    + word ngrams
    + character ngrams
    + emoticons
    - lexicons
    - negation
  '''

  num_allcaps = 0
  num_hashtags = 0
  num_elongated = 0
  for word in words:
    if word_is_all_caps:
      num_allcaps += 1
    if word[0] == '#':
      num_hashtags += 1
    if is_elongated(word):
      num_elongated += 1

  last_is_question_or_exclaim = 1 if contains_question_or_exclaim(words[-1]) else 0
  num_seq_question, num_seq_exclaim, num_seq_both = num_contiguous_question_exclaim(tweet)
  cluster_mem_vec = get_cluster_mem_vec(cids, w2c, words)

  pos_vec = get_pos_vec(pos.split())

  ngram_w_vec, ngram_n_vec, ngram_c_vec = get_ngram_vec(tweet, corpus_word_ng,
      corpus_nonc_ng, corpus_char_ng)

  emoticon_vec = get_emoticon_vec(orig, words[-1])

  features = [num_allcaps, num_hashtags, num_elongated,
      last_is_question_or_exclaim, num_seq_question, num_seq_exclaim,
      num_seq_both] + cluster_mem_vec + pos_vec + ngram_w_vec + ngram_n_vec +\
      ngram_c_vec + emoticon_vec
  return features


def add_arguments(parser):
  parser.add_argument('--input', type=str, required=True, help='Input file, which must be the output from Twokenize.')
  parser.add_argument('--clusters', type=str, required=True, help='The file containing the clusters data.')


def main(args):
  with open(args.input) as f:
    corpus = []
    for line in f:
      tok_tweet, pos, pos_conf, orig_tweet = line.strip().split('\t')
      corpus.append((tok_tweet, pos, pos_conf, orig_tweet))
    w2c, c2w, cids = load_clusters(args.clusters)
    word_ngrams, nonc_ngrams, char_ngrams = corpus_ngrams(corpus)
    for record in corpus:
      generate_features(record, w2c, cids, word_ngrams, nonc_ngrams, char_ngrams)
    print generate_features(corpus[2], w2c, cids, word_ngrams, nonc_ngrams, char_ngrams)


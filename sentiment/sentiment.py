import sklearn
import string
import tweetmotif

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

  # At first, all the sets are empty.
  for length in word_lengths:
    word_ngrams[length] = set()
    nonc_ngrams[length] = set()
  for length in char_lengths:
    char_ngrams[length] = set()

  for tweet in corpus:
    for length in char_lengths:
      for idx in range(0, len(tweet) - length + 1):
        char_ngrams[length].add(tweet[idx : idx+length])
    words = tweet.split()
    for length in word_lengths:
      for idx in range(0, len(words) - length + 1):
        ngram = words[idx : idx+length]
        word_ngrams[length].add(tuple(ngram))
        for j in range(0, length):
          tmp = list(ngram)
          tmp[j] = '*' # TODO: should this be a symbol like <*> ?
          nonc_ngrams[length].add(tuple(tmp))

  #print "words: %s \n\n nonc: %s \n\n char: %s \n\n" % (word_ngrams, nonc_ngrams, char_ngrams)
  return word_ngrams, nonc_ngrams, char_ngrams

def word_is_all_caps(word):
  for c in word:
    if c not in string.ascii_uppercase:
      return False
  return True

def is_elongated(word):
  elong_len = 3
  for idx in range(len(word) - elong_len + 1):
    if word[idx] == word[idx+1] and word[idx] == word[idx+2]:
      return True
  return False

def contains_emoticon(s):
  '''
    Checks if the passed-in string contains any emoticons.
    Adapted from a regex here: http://sentiment.christopherpotts.net/tokenizing.html
  '''
  pass # TODO

def generate_features(tweet):
  '''
    Takes in a tweet and generates a feature vector.
  '''
  words = tweet.split()

  '''
    Our features are: (+ done, / wip, - todo)
    / word ngrams
    / character ngrams
    + allcaps
    - pos
    + hashtags
    - lexicons
    - punctuation
    / emoticons
    + elongated words
    - clusters
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

def main():
  corpus = ["This is a tweet looong #happy #sad .", "This is 'nt a status message ."]
  word_ngrams, nonc_ngrams, char_ngrams = corpus_ngrams(corpus)
  print generate_features(corpus[0])


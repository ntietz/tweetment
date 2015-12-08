import argparse
import tweetment

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs the response generator code.')
  parser.add_argument('--classify', action='store_true', help='Run the classifier on input from stdin')
  parser.add_argument('--train', action='store_true', help='Run the tweet sentiment classifier trainer')
  args, uknowns = parser.parse_known_args()

  # The classifier will add any of the other arguments it needs.
  classifier = tweetment.SentimentClassifier(args, parser)

  if args.train:
    print 'Training sentiment classifier...'

    ## Reparse with the arguments we want now
    #tweetment.add_train_arguments(parser)
    #args = parser.parse_args()

    #tweetment.train(args)

  elif args.classify:
    classifier.classify()

  else:
    print 'No options matched.'


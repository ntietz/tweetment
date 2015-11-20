import argparse
import sentiment

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs the response generator code.')
  parser.add_argument('--classify', action='store_true', help='Run the classifier on input from stdin')
  parser.add_argument('--train', action='store_true', help='Run the tweet sentiment classifier trainer')
  args, uknowns = parser.parse_known_args()

  if args.train:
    print 'Training sentiment classifier...'

    # Reparse with the arguments we want now
    sentiment.add_arguments(parser)
    args = parser.parse_args()

    sentiment.main(args)

    exit(0)

  else:
    print 'No options matched.'


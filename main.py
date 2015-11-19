import argparse
import sentiment

parser = argparse.ArgumentParser(description='Runs the response generator code.')
parser.add_argument('--train-sentiment', action='store_true', help='Run the tweet sentiment classifier trainer')
args, uknowns = parser.parse_known_args()

if args.train_sentiment:
  print 'Training sentiment classifier...'

  # Reparse with the arguments we want now
  sentiment.add_arguments(parser)
  args = parser.parse_args()

  sentiment.main(args)

else:
  print 'No options matched.'


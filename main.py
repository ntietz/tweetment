import argparse
import tweetment

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs the response generator code.')
  parser.add_argument('--classify', action='store_true', help='Run the classifier on input from stdin')
  parser.add_argument('--train', action='store_true', help='Run the tweet sentiment classifier trainer')
  args, unknowns = parser.parse_known_args()

  if args.train:
    parser.add_argument('--input', type=str, required=True, help='Input directory')
    parser.add_argument('--clusters', type=str, required=True, help='The file containing the clusters data.')
    parser.add_argument('--cache', type=str, default='./cache', help='The directory we cache downloaded files in.')
    parser.add_argument('--savefile', type=str, default='./cache/model.pkl', help='The file we should save the model to.')
    args = parser.parse_args()

    print 'Training sentiment classifier...'

    classifier = tweetment.SentimentClassifier()
    classifier.train(args.input, args.cache, args.clusters, args.savefile)

  elif args.classify:
    parser.add_argument('--savefile', type=str, default='./cache/model.pkl', help='The file to load the model from.')
    parser.add_argument('--input', type=str, required=True, help='Input file')
    parser.add_argument('--output', type=str, required=True, help='Output file')
    args = parser.parse_args()

    classifier = tweetment.SentimentClassifier(args.savefile)

    classifier.classify_file(args.input, args.output)

  else:
    print 'No options matched.'


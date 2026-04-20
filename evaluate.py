import argparse
import logging
import collections
try:
    from sklearn.metrics import classification_report
except ImportError:
    print("For a more interesting output, please install the sklearn library")
    def classification_report(X, Y):
        assert len(X) == len(Y)

        stats = collections.Counter()
        for i in range(len(X)):
            if X[i] == Y[i]:
                stats["correct"] += 1
            else:
                stats["errors"] += 1

            # Optionally, We can also store the confusion matrix
            stats[ "Confusion {} vs {}".format(X[i],Y[i]) ] += 1
        logging.debug("\n".join(["{}: {}".format(k,v) for k,v in stats.items()]))

        results={}
        results["correct"] = stats["correct"]
        results["total"] = stats["correct"] + stats["errors"]
        results["accuracy"] = stats["correct"] / results["total"]
        return results

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="Evaluates the performance of a classification output")
    PARSER.add_argument("-v", action='store_true', default=False, help="Turns on verbose information")
    PARSER.add_argument('ref', help='Reference input file')
    PARSER.add_argument('test', help='Test input file')
    args = PARSER.parse_args()

    if args.v:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    data={"ref":[], "test":[]}

    with open(args.ref) as f:
        data["ref"] = [ line.split()[0] for line in f ]

    with open(args.test) as f:
        data["test"] = [ line.split()[0] for line in f ]

    print(classification_report(data["ref"], data["test"]))
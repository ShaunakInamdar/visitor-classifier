import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):

    evidence = []
    labels = []

    with open(filename, 'r') as file:
        lines = list(csv.DictReader(file))
        for line in lines:

            # convert month to a number
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for i in range(len(months)):
                if line['Month'] == months[i]:
                    line['Month'] = i
                    break

            # convert VisitorType to a number
            line['VisitorType'] = 1 if line['VisitorType'] == 'Returning_Visitor' else 0

            # convert Weekend to a number
            line['Weekend'] = 1 if line['Weekend'] == 'TRUE' else 0

            # convert fields to ints
            for field in ['Administrative', 'Informational', 'ProductRelated', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']:
                line[field] = int(line[field])

            # convert fields to floats
            for field in ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']:
                line[field] = float(line[field])

            evidence.append(list(line.values())[:-1])
            labels.append(1 if line['Revenue'] == 'TRUE' else 0)
    
    return (evidence, labels)

def train_model(evidence, labels):
    
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(evidence, labels)

    return classifier


def evaluate(labels, predictions):
    
    true_positives = {
        'labels': 0,
        'predictions': 0
    }

    true_negatives = {
        'labels': 0,
        'predictions': 0
    }

    for i in range(len(labels)):
        if labels[i] == 1:
            true_positives['labels'] += 1
            true_positives['predictions'] += predictions[i]
        else:
            true_negatives['labels'] += 1
            true_negatives['predictions'] += 1 - predictions[i]

    sensitivity = true_positives['predictions'] / true_positives['labels']
    specificity = true_negatives['predictions'] / true_negatives['labels']

    return (sensitivity, specificity)

if __name__ == "__main__":
    main()

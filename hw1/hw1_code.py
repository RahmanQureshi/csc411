import sklearn.feature_extraction
import sklearn.model_selection
import sklearn.tree
import numpy
import heapq
import graphviz
import math
from collections import defaultdict

vectorizer = sklearn.feature_extraction.text.CountVectorizer()
default_max_depths = range(1,5)
default_criterions = ['gini', 'entropy']

def log2(x):
    return math.log(x)/math.log(2)

def train_validate_test_split(X, Y):
    X_train, X_validate_test, Y_train, Y_validate_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3)
    X_validate, X_test, Y_validate, Y_test = sklearn.model_selection.train_test_split(X_validate_test, Y_validate_test, test_size=0.5)
    return X_train, X_validate, X_test, Y_train, Y_validate, Y_test

def visualize_tree(tree, name):
    data = sklearn.tree.export_graphviz(tree, out_file=None, feature_names=vectorizer.get_feature_names())
    graph = graphviz.Source(data)
    graph.render(name)

def load_data():
    headines = []
    with open('data/clean_fake.txt', 'r') as f:
        headlines = f.read().split('\n')
    n_fake = len(headlines)
    with open('data/clean_real.txt', 'r') as f:
        headlines.extend(f.read().split('\n'))
    n_real = len(headlines) - n_fake
    X = vectorizer.fit_transform(headlines)
    Y = [0 for i in range(0, n_fake)] + [1 for i in range(0, n_real)] # 0=> fake, 1=> real
    return train_validate_test_split(X, Y)

def validate(tree, X, Y):
    Y_predict = tree.predict(X)
    accuracy = float(sum([1 if Y[i]==Y_predict[i] else 0 for i in range(0, len(Y))]))/float(len(Y))
    return accuracy

def select_model(X, Y, X_validate, Y_validate, k=2, max_depths=default_max_depths, criterions=default_criterions):
    best_trees = []
    for depth in max_depths:
        for criteria in criterions:
            tree = sklearn.tree.DecisionTreeClassifier(criterion=criteria, max_depth=depth)
            tree.fit(X,Y)
            accuracy = validate(tree, X_validate, Y_validate)
            heapq.heappush(best_trees, (-accuracy, tree))
            print("Tree with max depth=%d, criterion=%s: validation_score=%f" % (depth, criteria, accuracy))
    trees = []
    for i in range(0,k):
        trees.append(heapq.heappop(best_trees))
    return trees

def labelCount(Y):
    """Returns a mapping from label to number of points in X with that label
    """
    Yset = set(Y)
    count = defaultdict(int)
    for i in range(0, len(Y)):
        count[Y[i]] += 1
    return count

def H(Y):
    label_to_count = labelCount(Y)
    total = 0
    for y in label_to_count:
        total += label_to_count[y]
    entropy = 0
    for y in label_to_count:
        p = float(label_to_count[y])/total
        entropy -= p*log2(p)
    return entropy

def splitX(X, Y, feature, split):
    """Returns a new X,Y based on split defined by the feature and split threshold
    """
    X_left = []
    Y_left = []
    X_right = []
    Y_right = []
    for i in range(0, len(X)):
        if X[i][feature] < split:
            X_left.append(X[i])
            Y_left.append(Y[i])
        else:
            X_right.append(X[i])
            Y_right.append(Y[i])
    return X_left, Y_left, X_right, Y_right

def get_feature_index(s):
    return vectorizer.vocabulary_.get(s)

def compute_information_gain(X, Y, feature, split):
    X = X.toarray()
    rootEntropy = H(Y)
    X_left, Y_left, X_right, Y_right = splitX(X, Y, feature, split)
    leftEntropy = H(Y_left)
    rightEntropy = H(Y_right)
    IG = rootEntropy - float(len(Y_left))/len(Y)*leftEntropy - float(len(Y_right))/len(Y)*rightEntropy 
    return IG

def main():
    X_train, X_validate, X_test, Y_train, Y_validate, Y_test = load_data()
    top_trees = select_model(X_train, Y_train, X_validate, Y_validate)
    print("Top trees:")
    print(top_trees)
    visualize_tree(top_trees[0][1], 'tree1')
    visualize_tree(top_trees[1][1], 'tree2')
    print("Top tree test set score=%f" % (validate(top_trees[0][1], X_test, Y_test)))
    print("Second tree test set score=%f" % (validate(top_trees[1][1], X_test, Y_test)))

    
    split_words = ['trump', 'korea', 'hillary', 'the', 'economic', 'and', 'election', 'america', 'clean', 'black']
    for word in split_words:
        featureIndex = get_feature_index(word)
        print("Information gain by splitting on %s:" % (vectorizer.get_feature_names()[featureIndex]))
        print(compute_information_gain(X_train, Y_train, featureIndex, 0.5))


if __name__ == '__main__':
    main()

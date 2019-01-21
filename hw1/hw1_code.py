import sklearn.feature_extraction
import sklearn.model_selection
import sklearn.tree
import numpy
import heapq

vectorizer = sklearn.feature_extraction.text.CountVectorizer()
default_max_depths = range(1,100)
default_criterions = ['gini', 'entropy']

def train_validate_test_split(X, Y):
    X_train, X_validate_test, Y_train, Y_validate_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3)
    X_validate, X_test, Y_validate, Y_test = sklearn.model_selection.train_test_split(X_validate_test, Y_validate_test, test_size=0.5)
    return X_train, X_validate, X_test, Y_train, Y_validate, Y_test

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
            print("Tree with max depth=%d, criterion=%s: accuracy=%f" % (depth, criteria, accuracy))
    trees = []
    for i in range(0,k):
        trees.append(heapq.heappop(best_trees))
    return trees
        
if __name__ == '__main__':
    X_train, X_validate, X_test, Y_train, Y_validate, Y_test = load_data()
    top_trees = select_model(X_train, Y_train, X_validate, Y_validate)
    print("Best trees:")
    print(top_trees)

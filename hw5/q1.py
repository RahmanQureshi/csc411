'''
Question 1 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
import scipy.stats
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    N = np.zeros(10) # count of number of classes
    for i in range(0, len(train_data)): # dangerous if too much train data
        k = train_labels[i]
        means[k] = means[k] + train_data[i]
        N[k] = N[k] + 1
    for i in range(0, len(means)):
        means[i] = means[i]/N[i]
    # Compute means
    return means

def compute_sigma_mles(train_data, train_labels, train_means):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    D = 64
    covariances = np.zeros((10, 64, 64))
    N = np.zeros(10)
    # Compute covariances
    for i in range(0, len(train_data)): # dangerous if too much train data
        k = int(train_labels[i])
        error = train_means[k] - train_data[i]
        error = error.reshape(64,1) # reshape into column vector
        covariances[k] = covariances[k] + error*error.transpose()
        N[k] = N[k] + 1
    for i in range(0, len(covariances)):
        covariances[i] = covariances[i]/N[i]
    for i in range(0, len(covariances)):
        covariances[i] = covariances[i] + 0.01*np.eye(D)
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 

    Student note: this was meant to be used in conditional_likelihood
    but I just computed it manually.
    '''
    return None

def multivariate_normal(x, mean, cov_inv, cov_det):
    d = len(x)
    error = x-mean
    error = error.reshape(d, 1) # reshape into column vec
    return (np.exp(-0.5*np.dot(np.dot(error.transpose(),cov_inv), error))/np.sqrt(cov_det*(2*np.pi)**d))[0][0]

def log_multivariate_normal(x, mean, cov_inv, cov_det):
    d = len(x)
    error = x-mean
    error = error.reshape(d, 1) # reshape into column vec
    return ((-0.5*np.dot(np.dot(error.transpose(),cov_inv), error)) - 0.5*np.log(cov_det*(2*np.pi)**d))[0][0]

def conditional_likelihood(digits, means, covariances, prior = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class

    Note: by bayes rule, p(y|x) = p(x|y) * p(y) / p(x). (y is the class).
    We ignore the denominator. 

    '''
    numClasses = len(means)
    n = len(digits)
    y = np.zeros((n,numClasses))
    cov_inv = np.zeros((10, 64, 64))
    cov_det = np.zeros(len(covariances))
    for i in range(0, len(covariances)):
        cov_inv[i] = np.linalg.inv(covariances[i])
        cov_det[i] = np.linalg.det(covariances[i])
    for i in range(0, n):
        for j in range(0, numClasses):
            y[i][j] = log_multivariate_normal(digits[i], means[j], cov_inv[j], cov_det[j]) + np.log(prior[j])
    return y

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances) # n data x 10 classes
    N = np.zeros(10) # count number of each class
    logy_avg = np.zeros((10, 10))
    for i in range(0, len(digits)):
        k = labels[i]
        logy_avg[k] = logy_avg[k] + cond_likelihood[i]
        N[k] = N[k] + 1
    for i in range(0, len(logy_avg)):
        logy_avg[i] = logy_avg[i] / N[i]
    return logy_avg

def accuracy(predictions, labels):
    assert(len(predictions) == len(labels))
    return float(sum([1 if predictions[i]==labels[i] else 0 for i in range(0, len(predictions))]))/float(len(predictions))

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)

def test_multivariate_functions():
    x = np.random.rand(5)
    mu = np.random.rand(5)
    cov = np.eye(5)
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    assert abs(scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=cov) - multivariate_normal(x, mu, cov_inv, cov_det)) < 1e-10
    assert abs(np.log(scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=cov)) - log_multivariate_normal(x, mu, cov_inv, cov_det)) < 1e-10

def eigen(A):
    """ Wrapper arround np.linalg.eig that additionally sorts the eigenvalues and eigenvectors
        with decending eigenvalue size (e.g. biggest evalue first).
    """
    evalues, evectors = np.linalg.eig(A)
    zipped = zip(evalues, evectors)
    zipped.sort(reverse=True)
    return [e[0] for e in zipped], [e[1] for e in zipped]

def run_tests():
    test_multivariate_functions()

def main():
    run_tests()

    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels, means)

    # Evaluation
    print("Average log likelihood for train:")
    y = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print(y)
    print("Average log likelihood for test:")
    ytest = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print(ytest)

    # Accuracy
    train_predictions = classify_data(train_data, means, covariances)
    train_accuracy = accuracy(train_predictions, train_labels)
    print("Accuracy on training set: %f" % train_accuracy)
    test_predictions = classify_data(test_data, means, covariances)
    test_accuracy = accuracy(test_predictions, test_labels)
    print("Accuracy on test set: %f" % test_accuracy)

    print("Computing eigenvalues")
    imgs = []
    for i, cov in enumerate(covariances):
        evalues, evectors = eigen(cov)
        print("Maximum evalue for %dth class: %f, second biggest: %f." % (i+1, evalues[0], evalues[1]))
        topevec = evectors[0]
        topevec = np.array([abs(e) for e in topevec])
        plt.subplot(2,5, i+1)
        plt.imshow(topevec.reshape(8,8))
        #plt.imshow(cov, cmap='Greys')
    plt.show()

if __name__ == '__main__':
    main()

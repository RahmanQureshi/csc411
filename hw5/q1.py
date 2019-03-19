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

def multivariate_normal(x, mean, cov, cov_inv, cov_det):
    d = len(x)
    error = x-mean
    error = error.reshape(d, 1) # reshape into column vec
    return (np.exp(-0.5*np.dot(np.dot(error.transpose(),cov_inv), error))/np.sqrt(cov_det*(2*np.pi)**d))[0][0]

def log_multivariate_normal(x, mean, cov, cov_inv, cov_det):
    d = len(x)
    error = x-mean
    error = error.reshape(d, 1) # reshape into column vec
    return (-0.5*np.dot(np.dot(error.transpose(),cov_inv), error) - 0.5*np.log(cov_det) - 0.5*d*np.log(2*np.pi))[0][0]
    #return np.log(scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=cov))

def generative_likelihood_not_log(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    numClasses = len(means)
    n = len(digits)
    pxgiveny = np.zeros((n,numClasses))
    cov_inv = np.zeros((10, 64, 64))
    cov_det = np.zeros(len(covariances))
    for i in range(0, len(covariances)):
        cov_inv[i] = np.linalg.inv(covariances[i])
        cov_det[i] = np.linalg.det(covariances[i])
    for i in range(0, n):
        for j in range(0, numClasses):
            pxgiveny[i][j] = multivariate_normal(digits[i], means[j], covariances[j], cov_inv[j], cov_det[j]) 
    return pxgiveny

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    numClasses = len(means)
    n = len(digits)
    logpxgiveny = np.zeros((n,numClasses))
    cov_inv = np.zeros((10, 64, 64))
    cov_det = np.zeros(len(covariances))
    for i in range(0, len(covariances)):
        cov_inv[i] = np.linalg.inv(covariances[i])
        cov_det[i] = np.linalg.det(covariances[i])
    for i in range(0, n):
        for j in range(0, numClasses):
            logpxgiveny[i][j] = log_multivariate_normal(digits[i], means[j], covariances[j], cov_inv[j], cov_det[j]) 
    return logpxgiveny

def conditional_likelihood(digits, means, covariances, prior = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class

    Note: by bayes rule, p(y|x) = p(x|y) * p(y) / p(x). (y is the class).

    p(x) = sum of p(x,k) over all k
    p(x) = sum of p(x|y=k)*p(y=k)
    This is logged.
    '''
    numClasses = len(means)
    n = len(digits)
    logpygivenx = np.zeros((n,numClasses))
    logpxgiveny = generative_likelihood(digits, means, covariances)
    pxgiveny = generative_likelihood_not_log(digits, means, covariances) # totally duplicate and unnecessary to calculate both of these
    logpx = np.zeros(n)
    for i in range(0, n):
        logpx[i] = np.log(np.dot(pxgiveny[i],prior))
    for i in range(0, n):
        for j in range(0, numClasses):
            logpygivenx[i][j] = logpxgiveny[i][j] + np.log(prior[j]) - logpx[i]
    return logpygivenx

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances) # n data x 10 classes
    N = 0 # count number of each class
    logy_avg = 0 
    for i in range(0, len(digits)):
        k = labels[i]
        logy_avg = logy_avg + cond_likelihood[i][k]
    logy_avg = logy_avg / float(len(labels))
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
    assert abs(scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=cov) - multivariate_normal(x, mu, cov, cov_inv, cov_det)) < 1e-10
    assert abs(np.log(scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=cov)) - log_multivariate_normal(x, mu, cov, cov_inv, cov_det)) < 1e-10

def top_eigen(A):
    """ Wrapper arround np.linalg.eig that additionally sorts the eigenvalues and eigenvectors
        with decending eigenvalue size (e.g. biggest evalue first).
    """
    evalues, evectors = np.linalg.eig(A)
    index = evalues.argmax();
    return evectors[:, index]

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
        top_evector = top_eigen(cov)
        plt.subplot(2,5, i+1)
        plt.imshow(top_evector.reshape(8,8))
        #plt.imshow(cov, cmap='Greys')
    plt.show()

if __name__ == '__main__':
    main()

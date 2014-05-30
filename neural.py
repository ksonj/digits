#!/usr/bin/env python2
# Implementation of a one layer neural network to identify handwritten digits
# (c) K. Jepsen, 2014
# kalle@siberie.de
# http://www.siberie.de/
# https://github.com/ksonj/

import numpy as np
from scipy import io
from scipy import optimize
import matplotlib as mpl
import matplotlib.pyplot
# Load data from matlab-style matrix-file
data = io.loadmat('ex4data1.mat')

# Put the input in X and the output in y
# Prepare data arrays
shape = np.shape(data['X'])
m = shape[0] # number of examples

X = data['X']
y = data['y']

def reshapeY(y,k=10):
    new_y = np.zeros((len(y),k))
    for i in range(len(y)):
        new_y[i,y[i]-1] = 1
    return new_y

def visualize(data, size, number=100):
# Visualize the data in "data", i.e. pick "number" random samples of size "size" and
# display them as grayscale image
    dim = int(number**0.5)
    # pick 100 examples from dataset
    idcs = np.random.randint(0,len(data),number)
    samples = data[idcs]
    disp = np.zeros(dim*np.array(size))
    for i in range(dim):
        for j in range(dim):
            img = np.reshape(samples[i*dim+j], size).T
            disp[i*size[0]:size[0]*(i+1), j*size[1]:size[1]*(j+1)] = img
    fig = mpl.pyplot.figure()
    ax = fig.add_subplot(111)
    dataImg = ax.imshow(disp, cmap='binary')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    #ax.imshow(np.reshape(samples[0],size), cmap='binary')
    mpl.pyplot.savefig('data.pdf')
    mpl.pyplot.close(fig)

def initWeights(n_in, n_hid, n_out):
# initialize weights
    # weights from input to hidden layer:
    epsilon = np.sqrt(6) / np.sqrt(n_input + n_hidden)
    #Theta1 = np.random.rand(n_hidden, n_input+1) #
    Theta1 = epsilon * (2*np.random.rand(n_hidden, n_input+1) - 1)

    # weights from hidden to output layer:
    epsilon = np.sqrt(6) / np.sqrt(n_hidden + n_output)
    #Theta2 = np.random.rand(n_output, n_hidden+1) * 2 * epsilon - epsilon#
    Theta2 = epsilon * (2*np.random.rand(n_output, n_hidden+1) - 1)
    return [Theta1, Theta2]

def sigmoid(z):
# Implement the sigmoid function
    return 1/(1 + np.exp(-1*np.array(z)))

def sigmoidGrad(z):
    return sigmoid(z) * (1 - sigmoid(z))

def cost(X, y, flatParams, n_in, n_hid, n_out, l=0):
# calculate the cost function to a given set of parameters
# flatParams is the flattened array of the parameters in Theta1 and Theta2
# n_in, n_hid and n_out are the sizes of the individual layers EXCLUDING the
# bias units
# l is the regularization parameter and defaults to 0 (i.e. no regularization)
    # iterate over each sample TODO: this can probably be vectorized in some way
    k = n_out # determine size of output layer
    m = len(X) # determine sample size

    # Retrieve original parameter matrices
    Theta1 = np.reshape(flatParams[:n_hid * (n_in+1)], (n_hid, n_in+1))
    Theta2 = np.reshape(flatParams[n_hid * (n_in+1):], (n_out, n_hid+1))
    gradTheta1 = np.zeros(np.shape(Theta1))
    gradTheta2 = np.zeros(np.shape(Theta2))
    Y = reshapeY(y, k)
    J = 0 # Cost
    for i in range(m):
        a1 = X[i]
        a1 = np.insert(a1,0,1) # add bias unit to input layer
        z2 = np.dot(Theta1,a1)
        a2 = sigmoid(z2)
        a2 = np.insert(a2,0,1) # add bias unit to hidden layer
        z3 = np.dot(Theta2,a2)
        a3 = sigmoid(z3)
        h = a3

        # Add cost
        J += 1./m * np.sum((-Y[i] * np.log(h) - (1 - Y[i]) * np.log(1 - h)))

        d3 = (a3 - Y[i])
        d2 = np.dot(Theta2.T[1:],d3) * sigmoidGrad(z2)
        gradTheta1 += np.dot(np.matrix(d2).T, np.matrix(a1))
        gradTheta2 += np.dot(np.matrix(d3).T, np.matrix(a2))

    # Add regularization to the cost function
    J += l/2./m * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))
    # Add regularization to the gradient
    gradTheta1[:,1:] += l*Theta1[:,1:]
    gradTheta2[:,1:] += l*Theta2[:,1:]
    # Flat out the gradient matrices for return
    gradFlat = 1./m * np.concatenate((gradTheta1.flatten(), gradTheta2.flatten()))
    return [J, gradFlat]

def gradientCheck(l=0):
    # Implements small neural network to check the gradients produced by the
    # backpropagation algorithm
    def testData(fan_out, fan_in):
        W = np.zeros((fan_out, fan_in+1))
        W = np.reshape(np.sin(range(fan_out*(fan_in+1))), np.shape(W)) / 10.
        return W

    def numericGradient(F, params):
        numgrad = np.zeros(np.shape(params))
        perturb = np.zeros(np.shape(params))
        e = 1e-4
        for p in range(len(params)):
            perturb[p] = e
            sPlus = F(params + perturb)[0] # use the cost
            sMinus = F(params - perturb)[0] # use the cost
            numgrad[p] = (sPlus - sMinus) / (2*e)
            perturb[p] = 0
        return numgrad

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3 # output layer size
    m = 5 # number of samples

    # Generate some test data
    Theta1 = testData(hidden_layer_size, input_layer_size)
    Theta2 = testData(num_labels, hidden_layer_size)

    X = testData(m, input_layer_size - 1)
    y = 1 + np.mod(range(m), num_labels)

    # Unroll parameters
    flatParams = np.concatenate((Theta1.flatten(), Theta2.flatten()))

    # Shorthand, so you only need to provide the parameters, the rest stays the
    # same
    def costFunc(params):
        return cost(X, y, params, input_layer_size, hidden_layer_size,
                num_labels, l)

    # Compute gradients w/ backpropagation
    [propcost, propgrad] = costFunc(flatParams)
    # Numerically compute gradients
    numgrad = numericGradient(costFunc, flatParams)

    return (np.linalg.norm(numgrad - propgrad,2) / np.linalg.norm(numgrad +
        propgrad,2))

def predict(params, n_in, n_hid, n_out, X):
    Theta1 = np.reshape(params[:n_hid * (n_in+1)], (n_hid, n_in+1))
    Theta2 = np.reshape(params[n_hid * (n_in+1):], (n_out, n_hid+1))
    m = len(X)
    p = np.zeros(m)
    # Add bias units to input
    Xp = np.concatenate((np.ones((m,1)), X), axis=1)
    h1 = sigmoid( np.dot(Xp, Theta1.T) )
    # Add bias unit
    h1 = np.concatenate((np.ones((m,1)), h1), axis=1)
    h2 = sigmoid( np.dot(h1, Theta2.T) )

    # return the index of the maximum value along axis 1
    return np.argmax(h2, axis=1)

def testPerformance(params, n_in, n_hid, n_out, X, y):
    p = predict(params, n_in, n_hid, n_out, X)
    p = np.matrix(p) + 1
    acc = np.sum(y.T == p)
    return 100. * acc/len(X)
    # Test the performance of the network, i.e. predict the values given a set
    # of parameters and inputs and compare to the desired outputs

def dispProgress(xk):
    global nEval
    nEval += 1
    print("Iteration %i" % nEval)
    # Callback function to be called after each iteration of the optimization
    # process to display the current status

nEval = 0
# Determine number of input and output units
n_input = np.shape(X)[1] # excl. bias unit
n_output = 10 #TODO: this has to be determined first, how?
# define size of hidden layer excl. bias unit
n_hidden = 25

######## For debugging purposes, load the given weights from the excercise #####
idealWeights = io.loadmat('ex4weights.mat')
Theta1 = idealWeights['Theta1']
Theta2 = idealWeights['Theta2']

params = np.concatenate((Theta1.flatten(), Theta2.flatten()))

[J, grad] = cost(X, y, params, n_input, n_hidden,  n_output, l=0)
print("J = %.6f without regularization" % J)
l = 1
[J, grad] = cost(X, y, params, n_input, n_hidden,  n_output, l=l)
print("J = %.6f with regularization (l = %.2f)" % (J, l))

# Reshape gradients to original shapes
gradTheta1 = np.reshape(grad[:n_hidden*(n_input+1)], (n_hidden, n_input+1))
gradTheta2 = np.reshape(grad[n_hidden*(n_input+1):], (n_output, n_hidden+1))

# Wrap the objective function that returns the cost and the gradient (cost), so
# that it only takes two arguments (i.e. the initial guess for the weights and
# the regularization parameter)
def wrappedCost(params):
    return cost(X, y, params, n_input, n_hidden, n_output, 1)
    #return cost(X, y, params, n_input, n_hidden, n_output, l)

####### Here the actual program starts ########
# Initialize weights
[Theta1, Theta2] = initWeights(n_input, n_hidden, n_output)
# Unroll them into a one dimensional vector
initParams = np.concatenate((Theta1.flatten(), Theta2.flatten()))
# Set regularization parameter
l = 1
# Set the maximum number of iterations
maxiter=1000
# Compute initial cost
[J, grad] = cost(X, y, initParams, n_input, n_hidden,  n_output, l=l)
print("Initial cost is J = %.6f" % (J))
# Start the optimization
res = optimize.minimize(wrappedCost, initParams, method='CG',
        jac=True, options={'maxiter': maxiter, 'disp': True}, callback=dispProgress)

# Retrieve the optimal values for Theta1 and Theta1
Theta1 = np.reshape(res.x[:n_hidden * (n_input+1)], (n_hidden, n_input+1))
Theta2 = np.reshape(res.x[n_hidden * (n_input+1):], (n_output, n_hidden+1))

# Save these parameters to disk
np.savez('bestTheta', Theta1, Theta2)
print("Parameters saved to bestTheta.npz")
# Evaluate the accuracy of the prediction
perf = testPerformance(res.x, n_input, n_hidden, n_output, X, y)
print("The neural network has %.2f%% accuracy on the training set" % (perf))
# Program flow:
#visualize(X[:,1:], (20,20))


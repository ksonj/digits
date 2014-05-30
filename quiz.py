#!/usr/bin/env python2

# This script demonstrates how well the neural network predicts the numbers
# given
# (c) K. Jepsen, 2014
# kalle@siberie.de
# http://www.siberie.de/
# https://github.com/ksonj/

import numpy as np
from scipy import io
import matplotlib as mpl
import matplotlib.pyplot

def sigmoid(z):
    return 1./(1 + np.exp(-1*np.array(z)))

data = io.loadmat('ex4data1.mat')
X = data['X']
y = data['y']

# Load weights:
weights = np.load('bestTheta.npz')

Theta1 = weights['arr_0']
Theta2 = weights['arr_1']

#idealWeights = io.loadmat('../ex4weights.mat')
#Theta1 = idealWeights['Theta1']
#Theta2 = idealWeights['Theta2']

# Determine network size
n_in = np.shape(X)[1]
n_hid = np.shape(Theta1)[0]
n_out = np.shape(Theta2)[0]

def predict(x):
    # Takes a sample and computes the output of the network
    # Add bias unit
    xp = np.concatenate(([1], x))
    h1 = sigmoid(np.dot(xp, Theta1.T))
    #return h1
    h1 = np.concatenate(([1], h1))
    h2 = sigmoid(np.dot(h1, Theta2.T))
    p = np.argmax(h2)+1
    if p == 10:
        p = 0
    return p

fig = mpl.pyplot.figure()
ax = fig.add_subplot(111)
mpl.pyplot.ion()
mpl.pyplot.show()

while True:
    # Get a random number
    i = np.random.randint(0,5000)
    x = X[i]
    ax.set_title('This looks like a %i' % predict(x), size=24)
    xImg = np.reshape(x,(20,20)).T
    img = ax.imshow(xImg, cmap='binary')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.canvas.draw()
    raw_input("Press Enter to continue...")


#!/usr/bin/env python2
# This programs visaulizes the hidden units of the neural network
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot

weights = np.load('bestTheta.npz')

Theta1 = weights['arr_0']
Theta2 = weights['arr_1']

fig = mpl.pyplot.figure()
ax = fig.add_subplot(111)
dim = 5
size = (20,20)
disp = np.zeros(dim*np.array(size))
for i in range(dim):
    for j in range(dim):
        unit = np.reshape(Theta1[i*dim+j,:-1],size).T
        disp[i*size[0]:size[0]*(i+1), j*size[1]:size[1]*(j+1)] = unit
i = 0
img = ax.imshow(disp, cmap='PuOr')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
mpl.pyplot.show()

#Identification of handwritten digits
This is a Python-reimplementation of programming exercise 4 from the Coursera
course on [machine learning](https://class.coursera.org/ml-004). The implemented neural network identifies handwritten digits.

##Dependencies
* [scipy](http://www.scipy.org/)
* [matplotlib](http://matplotlib.org/)

##Files
`neural.py` computes the weights from the data in `ex4data1.mat`.
`quiz.py` presents the performance of the classification. 
`showFeatures.py` visualizes some features of the data.

`bestTheta.npz` contains the computed weights. This file is overwritten when running `neural.py`.

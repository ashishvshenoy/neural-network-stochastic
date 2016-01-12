# neural-network-stochastic
## A single-layer neural network using stochastic gradient descent. ##
* This is an implementation of an algorithm that learns a single-layer neural network using stochastic gradient descent (on-line training). 
* This algorithm is intended for binary classification problems where the output unit used a sigmoid function.
* Stochasic gradient descent is used to minimize squared error.
* Test the obtained neural net on a data set that represented energy within particular frequency bands when the signal from a RADAR bounces off a given object. This data was used to determine if the object is a rock or a mine. 
* The input format of the training file is expected to be in ARFF. http://www.cs.waikato.ac.nz/ml/weka/arff.html
* This program does stratified cross validation with the specified number of folds
* Usage `neuralnet <data-set-file> n l e`
<br> where n = number of folds in cross validation, l = learning rate, e = number of training epochs.

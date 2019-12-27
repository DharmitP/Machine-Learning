# Machine-Learning
 Assignments #2 and #3 for Intro to Machine Learning Course
 
## Assignment 2
### Part 1
An implementation of a neural network classifier using pure Numpy only. There are 3 layers: 
- input layer
- hidden layer with ReLu activation
- output layer with Softmax

Cross-entropy loss is calculated and Gradient Descent with momentum is used for optimization. Backpropogation calculations are done manually through pure Numpy.

### Part 2
An implementation of a Convolutional Neural Network (CNN) using Tensorflow 2.0 and Keras. CNN consists of:
- input layer
- 3x3 convolutional layer with 32 filters (strides of 1 with Xavier initialization for each filter)
- batch normalization layer
- 2x2 max pooling layer
- flattening layer
- fully connected layer
- ReLU activation
- fully connected layer
- softmax output

Cross-entropy loss is calculated and the model is optimized using Stochastic Gradient Descent by utilizing the Adam optimizer in Tensorflow. Effects of L2 normalization is observed as well as dropout to control overfitting.

## Assignment 3
### Part 1
K-means algorithm is implemented to learn clusters in a 2D dataset. The model is optimized by minmizing the loss function using the Adam optimizer. Optimal number of clusters are also explored.

### Part 2
Mixtures of Gaussians (MoG) is implemented to learn the clusters for a 2D and 100D dataset. The model optimizes the log-liklihood loss function. Differences in learning for validation loss and training loss is also explored.


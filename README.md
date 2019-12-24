# multiclass_classifiers
A selection of multi-class classifiers to predict the orientation of an image vector.

For training, the program should be run like this:
./orient.py train train_file.txt model_file.txt [model]

For testing, the program should be run like this:
./orient.py test test_file.txt model_file.txt [model]

options for train_file.txt - train-data.txt
options for test_file.txt - test-data.txt
options for model_file.txt - knn_model.txt, tree_model.txt and nnet_model.txt
options for [model] - nearest (for k-nearest neighbors), tree (for Decision tree) and nnet (for Neural network)

All the above classifiers are made without using Tensorflow, PyTorch, Scikit-learn or any other convenient library.

Images are rescaled to a very tiny "micro-thumbnail" of 8 × 8 pixels, resulting in an 8 × 8 × 3 = 192 dimensional feature vector.
These vectors are stored as space separated lines in .txt files.

The text files viz. train-data.txt and test-data.txt have one row per image, where each row is the feature vector formatted like:
photo_id correct_orientation r11 g11 b11 r12 g12 b12 ...
where:
• photo id is a photo ID for the image. Can be verified at http://www.flickr.com/photo_zoom.gne?id=photo_id
• correct orientation is 0, 90, 180, or 270.
• r11 refers to the red pixel value at row 1 column 1, r12 refers to red pixel at row 1 column 2, etc.,
each in the range 0-255.


----------

## KNN

### Implementation -
The concept of KNN is really simple - assign any given point to the same category as its nearest neighbours. The main challange for this problem was the running time. As for every test example, we need to iterate over some 3600 rows and calculate the distance for every single one. We tried to minimize the time for this distance calculation by trying several approaches such as storing the images in various ndarray formats and using different functions to calculate the Eculidean distance. The reason we chose Eculidean distance was because it performed significantly better the Manhattan distance we tried calculating using scipy library (Used for only our internal testing and cross-verification. The submitted code does not use scipy.) Another speed up was writing the model to the file for which we used "Pickle" as we observed it was faster than other techniques. We also tried changing the training data size to see its effect on the performance.

### Results -
The accuracy of the k-nearest classifier does not change "much" with k, for k greater than 10.

| Value of k   | Accuracy | Time (in seconds) |
|--------------|----------|-------------------|
|       5      |   69.14  |        193        |
|      10      |   70.31  |        194        |
|      50      |   71.26  |        202        |
|      100     |   70.31  |        203        |
|      900     |   70.73  |        206        |

As can it can be seens from above data, running time of the algorithm also does not change much by changing the value of k.
Because finding the k-nearest points takes the most amount of time from the total number of datapoints.
Sorting and comparing the k points takes comparatively little time.

We would recommend K-Nearest Neighbors for the kinds of classification where we don't have a lot of data and there is
little correlation between features and labels. KNN is also useful when the training data is changing fast and it is a requirement
that the newer values are taken into account. Updating the model for KNN is the fastest among the other three classifiers. As KNN does not reqiure any specific training tasks it is perfect for a classification scenario where you can't observe all the data at once.

Training performance does not change much with change training dataset size (after we have a certain threshold of data available) because for KNN, all that is required is to store the training data as it is in the model or only change the "way" data is stored.  


## Decision Tree

In order to make a decision tree, we need to find a best split point (also called a threshold) of a feature to split the data into 2
branches of a binary tree. To decide the effectivenes of a split point we use a metric called gini coefficient.
In this algorithm we are selecting 'N' features randomly out of the 192 features given to us. We then find the best split
points from each of the features by trying out the unique elements in that feature-list (usually between 0 to 255 for first few iterations) as split points. These 'N' features are essentially column indices.

Table:-

| Number of features (N) | Accuracy | Time   |
|------------------------|----------|--------|
|            5           |   54.08  | 10 min |
|           10           |   61.29  | 27 min |
|           20           |   62.78  | 50 min |


We would recommend this classifier if the potential client wants very fast prediction. As compared to any other classifier, the decision
tree provides fastest classification with the trade off that the accuracy would be comparatively lower even though the training times may be large. The training performance is directly proportional to the number of samples (almost linear).


## Neural Networks
Using backpropogation algorithm, we update the weights assigned to each neuron.
Steps:

Read the file
Initialise
  Assign random weights to the neurons in hiden layer and output layer
Forward Propogation
  First, activate the neurons by adding the dot product of the weights and inputs to the bias
  Then pass the values through a sigma function. (tried using softmax as well, but implemented sigma)
Back propogation
  Find the error by subtracting expected value from the output calculated.
  Update the weights : updated weights = alpha(weight-derivative of E wrt W)
  To calculate the derivative, we use chain rule. dE/dW = dE/dX.dX/dW
  dX/dW equates to Y. So dE/dW = dE/dX.Y
  dE/dX= dY/dX.dE/dY
  d(f(X))/dx.dE/dY = f(x)[1-f(x)].dE/dY
  All the terms are dependent on y, thus we can calculate the derivative and hence update the weights accordingly.
This is training our neural network. Once it is fully trained, we test it using test data.

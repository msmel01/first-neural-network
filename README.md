# Digit Recognition Neural Network

This neural network can recognize handwritten digits from the MNIST dataset.

The smaller train and test set were obtained from [here](https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_test_10.csv) and [here](https://raw.githubusercontent.com/makeyourownneuralnetwork/makeyourownneuralnetwork/master/mnist_dataset/mnist_train_100.csv).

To run the code, use `python neural_net_class.py`.

To train and train on the entire dataset:
* Download the [training set](http://www.pjreddie.com/media/files/mnist_train.csv) and the [test set](http://www.pjreddie.com/media/files/mnist_test.csv) and store them in your desired location.
* Change line 70 to `data_file = open('path/to/the/training/csv', 'r')`.
* Change line 83 to `test_data_file = open('path/to/the/test/csv', 'r')`.
* Run the code as before!

Reference material: Make Your Own Neural Network by Tariq Rashid
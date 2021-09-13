import numpy as np
import scipy.special
import matplotlib.pyplot

class NeuralNet:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # initial weight matrices randomly picked from -0.5 to 0.5
        # self.input_to_hidden_weights = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        # self.hidden_to_output_weights = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

        # sophisticated weight initialization from normal distribution with mean 0 and standard dev
        # 1/sqrt(num of incoming links)
        self.input_to_hidden_weights = np.random.normal(0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.hidden_to_output_weights = np.random.normal(0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        # sigmoid activation function
        self.activation_fuction = lambda x: scipy.special.expit(x)


    def train(self, input_list, target_list):
        input = np.array(input_list, ndmin=2).T
        target = np.array(target_list, ndmin=2).T

        hidden_layer_input = np.dot(self.input_to_hidden_weights, input)
        hidden_layer_output = self.activation_fuction(hidden_layer_input)

        output_layer_input = np.dot(self.hidden_to_output_weights, hidden_layer_output)
        output_layer_output = self.activation_fuction(output_layer_input)

        # error is (target - actual)
        output_layer_error = target - output_layer_output

        # errors of hidden layer nodes
        hidden_layer_error = np.dot(self.hidden_to_output_weights.T, output_layer_error)

        # adjust weights of the hidden layer
        delta_hidden_weights = self.learning_rate * np.dot(output_layer_error * output_layer_output * (1 - output_layer_output), np.transpose(hidden_layer_output))
        self.hidden_to_output_weights += delta_hidden_weights

        # adjust weight of the input layer
        delta_input_weights = self.learning_rate * np.dot(hidden_layer_error * hidden_layer_output * (1 - hidden_layer_output), np.transpose(input))
        self.input_to_hidden_weights += delta_input_weights


    def query(self, input_list):
        input = np.array(input_list, ndmin=2).T

        hidden_layer_input = np.dot(self.input_to_hidden_weights, input)
        hidden_layer_output = self.activation_fuction(hidden_layer_input)

        output_layer_input = np.dot(self.hidden_to_output_weights, hidden_layer_output)
        output_layer_output = self.activation_fuction(output_layer_input)

        return output_layer_output


if __name__ == '__main__':
    input_nodes =  784 # image is 28 by 28
    output_nodes = 10
    hidden_nodes = 100
    learning_rate = 0.3

    net = NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

    data_file = open('./mnist_train_100.csv', 'r')
    data_list = data_file.readlines()
    data_file.close()

    for number in data_list:
        all_values = number.split(',')
        scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # preparing the training data
        output_nodes = 10
        target_output = np.zeros(output_nodes) + 0.01
        target_output[int(all_values[0])] = 0.99
        net.train(scaled_input, target_output)

    test_data_file = open('./mnist_test_10.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # testing neural network
    score = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print('Correct label is {}'.format(correct_label))
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = net.query(inputs)
        label = np.argmax(outputs)
        print('Network predicts {}'.format(label))
        if (label == correct_label):
            score.append(1)
        else:
            score.append(0)
        print('\n')
    
    scorecard_array = np.asarray(score)
    performance_score = scorecard_array.sum() / scorecard_array.size
    print('performance score is {}'.format(performance_score))
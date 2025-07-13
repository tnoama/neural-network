import sys
import os
import time

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork
import part1.Utils as Utils
import Jac_test
import matplotlib.pyplot as plt
import part1.Grad_test as grad_test

if __name__ == "__main__":

        # 2.1
        Jac_test.jac_test_layer(2, 3, "W")
        Jac_test.jac_test_layer(2, 3, "b")

        # 2.2
        Jac_test.jac_test_resnet_layer(5, "W1")
        Jac_test.jac_test_resnet_layer(5, "W2")
        Jac_test.jac_test_resnet_layer(5, "b")
        
        # 2.3
        train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/PeaksData.mat")
        learning_rate = 0.1
        activation = 'TanH'
        resNet = False
        hidden_layer = [10, 10]
        model_layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]

        model = NeuralNetwork(model_layers, activation, resNet)
        data_sample = np.array([train_data[0]])
        label_sample = np.array([train_labels[0]])
        grad_test.gradient_test_NN(model, data_sample, label_sample, "Gradient test for NN")

        # 2.4 Network lengths experiments
        data_sets = ["Datasets/PeaksData.mat"]
        hidden_layers = [
                        [],
                        [10],
                        [10, 10, 10],
                        [10, 10, 10, 10, 10],
                        [50],
                        [50, 50, 50]
                        ]
        learning_rates = [0.1, 0.01, 0.001]
        batch_sizes = [32, 64, 128]
        epochs = 200
        is_resNet = False

        for data_set in data_sets:
                train_data, train_labels, val_data, val_labels = Utils.load_data(data_set)
                for hidden_layer in hidden_layers:
                        layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]
                        for learning_rate in learning_rates:
                                for batch_size in batch_sizes:
                                        model = NeuralNetwork(layers, 'ReLU', is_resNet)
                                        start_time = time.time()
                                        _,_,_,val_accuracy = model.train(train_data, train_labels, val_data, val_labels, batch_size, epochs, learning_rate)
                                        end_time = time.time()
                                        elapsed_time = end_time - start_time
                                        print(f"Data set: {data_set}, Hidden layers: {hidden_layer}, Learning rate: {learning_rate}, "
                                                f"Batch size: {batch_size}, Accuracy: {val_accuracy[-1]}, "
                                                f"Training time: {elapsed_time:.2f} seconds")

        # 2.5 Network lenghts with paramerter constraints
        epochs = 200
        activation = 'ReLU'
        batch_size = 32
        learning_rate = 0.1
        
        hidden_layers500 = [
                        [45],
                        [17, 17],
                        [5, 10, 12, 15],
                        [15, 12, 10, 5],
                        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
                        ]
        hidden_layers500_resnet = [
                        [13, 13],
                        [9, 9, 9],
                        [5, 5, 5, 5, 5, 5, 5, 5, 5]
                        ]

        train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/PeaksData.mat")
        for hidden_layer in hidden_layers500:
                layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]
                model = NeuralNetwork(layers, activation, False)
                _,_,_,val_accuracy = model.train(train_data, train_labels, val_data, val_labels, batch_size, epochs, learning_rate)
                print(f"Data set: PeaksData , resNet: False, layers: {[2] + hidden_layer + [2]}, Params: {Utils.calculate_total_params(layers, False)}, accuracy: {val_accuracy[-1]}")
        for hidden_layer in hidden_layers500_resnet:
                layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]
                model = NeuralNetwork(layers, activation, True)
                _,_,_,val_accuracy = model.train(train_data, train_labels, val_data, val_labels, batch_size, epochs, learning_rate)
                print(f"Data set: PeaksData , resNet: True, layers: {[2] + hidden_layer + [2]}, Params: {Utils.calculate_total_params(layers, True)}, accuracy: {val_accuracy[-1]}")

        train_data, train_labels, val_data, val_labels = Utils.load_data("Datasets/GMMData.mat")
        for hidden_layer in hidden_layers500:
                layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]
                model = NeuralNetwork(layers, activation, False)
                _,__cached__,_,val_accuracy = model.train(train_data, train_labels, val_data, val_labels, batch_size, epochs, learning_rate)
                print(f"Data set: GMMData , resNet: False, layers: {[5] + hidden_layer + [5]}, Params: {Utils.calculate_total_params(layers, False)}, accuracy: {val_accuracy[-1]}")
        for hidden_layer in hidden_layers500_resnet:
                layers = [train_data.shape[1]] + hidden_layer + [len(np.unique(train_labels))]
                model = NeuralNetwork(layers, activation, True)
                _,_,_,val_accuracy = model.train(train_data, train_labels, val_data, val_labels, batch_size, epochs, learning_rate)
                print(f"Data set: GMMData , resNet: True, layers: {[5] + hidden_layer + [5]}, Params: {Utils.calculate_total_params(layers, True)}, accuracy: {val_accuracy[-1]}")
import math
import random

# Randomly initialize weights and biases for hidden and output layers
def initialize_weights(input_size, hidden_size, output_size):
    weights_matrix_1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
    bias_vector_1 = [random.uniform(-1, 1) for _ in range(hidden_size)]

    weights_matrix_2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
    bias_vector_2 = [random.uniform(-1, 1) for _ in range(output_size)]

    return (weights_matrix_1, bias_vector_1, weights_matrix_2, bias_vector_2)

# Activation function for hidden/output layers
def sigmoid(z):
    return 1 / ( 1 + math.exp(-z) )

# Computes derivative for back propogation
def sigmoid_derivative(z):
    return z * ( 1 - z )

# Compute hidden activations and output
def forward_pass(X, W1, b1, W2, b2):
    raw_inputs = [
        (X[0] * W1[i][0] + X[1] * W1[i][1] + X[2] * W1[i][2] + X[3] * W1[i][3] + b1[i]) 
        for i in range(len(W1))
        ]

    hidden_activation = [sigmoid(z) for z in raw_inputs]

    output = hidden_activation[0] * W2[0][0] + hidden_activation[1] * W2[0][1] + b2[0]
    y_pred = sigmoid(output)

    return y_pred, hidden_activation

two_by_two_vectors = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
]

majority = [
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1
]

W1, b1, W2, b2 = initialize_weights(4, 2, 1)
forward_pass(two_by_two_vectors[0], W1, b1, W2, b2)
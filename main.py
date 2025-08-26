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

def compute_loss(y_true, y_pred):
    return (1 / 2) * math.pow((y_true - y_pred), 2)

def backward_pass(X, y_true, y_pred, hidden_activation, W1, b1, W2, b2):
    output_error = (y_pred - y_true) * y_pred * (1 - y_pred)
    dW2 = [output_error * h for h in hidden_activation]
    db2 = output_error * 1

    hidden_error = [output_error * W2[0][i] * h* (1 - h) for i, h in enumerate(hidden_activation)]
    dW1 = [[hidden_error[i] * x for x in X] for i in range(len(hidden_error))]
    db1 = hidden_error

    return dW1, db1, dW2, db2    

def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    for i in range(len(W1)):
        for j in range(len(W1[i])):
            W1[i][j] -= learning_rate * dW1[i][j]

    for i in range(len(b1)):
        b1[i] -= learning_rate * db1[i]

    for i in range(len(W2[0])):
        W2[0][i] -= learning_rate * dW2[i]

    b2[0] -= learning_rate * db2

    return W1, b1, W2, b2

def train(X, Y, epochs, learning_rate):
    W1, b1, W2, b2 = initialize_weights(4, 2, 1)

    for epoch in range(epochs):
        for x, y in zip(X, Y):
            y_pred, hidden_activation = forward_pass(x, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward_pass(x, y, y_pred, hidden_activation, W1, b1, W2, b2)
            W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if epoch % 100 == 0:
            y_pred_epoch, _ = forward_pass(x, W1, b1, W2, b2)
            loss = (y_pred_epoch - y)**2 / 2
            print(f"Epoch {epoch}, Loss {loss}")

    return W1, b1, W2, b2

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

train(two_by_two_vectors, majority, 1000, 0.2)

import random

# Randomly initialize weights and biases for hidden and output layers
def initialize_weights(input_size, hidden_size, output_size):
    weights_matrix_1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
    bias_vector_1 = [random.uniform(-1, 1) for _ in range(hidden_size)]

    weights_matrix_2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
    bias_vector_2 = [random.uniform(-1, 1) for _ in range(output_size)]

    return (weights_matrix_1, bias_vector_1, weights_matrix_2, bias_vector_2)

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

initialize_weights(4, 2, 1)
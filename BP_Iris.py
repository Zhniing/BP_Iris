from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

attributes = iris.data
target = iris.target  # class label: 0 1 2
label = np.zeros((150, 3))
# regular the target into label(output vector)
for j in range(len(target)):
    if target[j] == 0:
        label[j] = np.array([1, 0, 0])
    elif target[j] == 1:
        label[j] = np.array([0, 1, 0])
    elif target[j] == 2:
        label[j] = np.array([0, 0, 1])
# feature_names = iris.feature_names


# Size of BP

input_dim = 4  # dimension of input
output_dim = 3  # dimension of output
depth = 3  # depth of neural network
units_per_hidden = 10  # units of per hidden layer

# Parameters of BP

# w_input = np.zeros((units_per_hidden, input_dim))  # weight of input to hidden
w_input = -1 + 2 * np.random.random((units_per_hidden, input_dim))  # weight of input to hidden
# w_hidden = np.zeros((depth-2, units_per_hidden, units_per_hidden))  # weight of hidden to hidden
w_hidden = -1 + 2 * np.random.random((depth - 2, units_per_hidden, units_per_hidden))  # weight of hidden to hidden
# w_output = np.zeros((output_dim, units_per_hidden))  # weight of hidden to output
w_output = -1 + 2 * np.random.random((output_dim, units_per_hidden))  # weight of hidden to output

b_hidden = -1 + 2 * np.random.random((depth - 1, units_per_hidden))  # bias of hidden
b_output = -1 + 2 * np.random.random(output_dim)  # bias of output

# Record the output of every hidden layer
output_hidden = np.zeros((depth - 1, units_per_hidden))

# Record the partial differential of the loss function for each neuron input
differential_hidden = np.zeros((depth - 1, units_per_hidden))


# differential_output = np.zeros(output_dim)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


#  For a sample，run a forward propagation，return the output value
def forward_propagation(train_sample):
    z = np.dot(w_input, train_sample) + b_hidden[0]
    o = sigmoid(z)
    output_hidden[0] = o
    for i in range(depth - 2):
        z = np.dot(w_hidden[i], o) + b_hidden[i + 1]
        o = sigmoid(z)
        output_hidden[i + 1] = o
    z = np.dot(w_output, o).reshape(output_dim, 1) + b_output.reshape(output_dim, 1)
    # o = sigmoid(z)
    z.resize(z.size)
    result = softmax(z)
    return result


# Loss function: cross entropy
def backward_propagation(diff_output):
    num_hidden = w_hidden.shape[0] + 1  # the number of hidden layers
    global differential_hidden
    # last hidden layer
    differential_hidden[-1] = \
        np.dot(
            np.dot(
                np.transpose(w_output), diff_output
            ),
            np.dot(
                output_hidden[num_hidden - 1], 1 - output_hidden[num_hidden - 1]
            )
        )
    # other hidden layer
    for i in range(num_hidden - 1):
        index = -(i + 2)
        differential_hidden[index] = \
            np.dot(
                np.dot(
                    np.transpose(w_hidden[index + 1]), differential_hidden[index + 1]
                ),
                np.dot(
                    output_hidden[index], [1] + [-1] * output_hidden[index]
                )
            )


# For a sample, update the parameters
def update_parameters(x):
    lr = 0.01  # learning rate
    # declare global variable
    global w_input
    # global b_hidden
    global w_output
    global b_output

    # d = differential_hidden[0]
    # d.shape = [10, 1]
    # x.shape = [1, 4]
    w_input = w_input - lr * np.dot(differential_hidden[0].reshape(10, 1), x.reshape(1, input_dim))
    b_hidden[0] = b_hidden[0] - lr * differential_hidden[0]

    num_hidden = w_hidden.shape[0] + 1  # the number of hidden layers
    for i in range(num_hidden - 1):
        w_hidden[i] = w_hidden[i] - lr * np.dot(differential_hidden[i + 1], output_hidden[i])
        b_hidden[i + 1] = b_hidden[i + 1] - lr * differential_hidden[i + 1]

    differential_output.shape = [output_dim, 1]
    o = output_hidden[num_hidden - 1]
    o.shape = [1, 10]
    w_output = w_output - lr * np.dot(differential_output, o)
    b_output = b_output.reshape(output_dim, 1) - lr * differential_output


# Main
t1 = forward_propagation(attributes[10])
print('t1', t1)

for n in range(200):
    for k in range(attributes.shape[0]):
        differential_output = forward_propagation(attributes[k]) - label[k]
        backward_propagation(differential_output)
        update_parameters(attributes[k])

t2 = forward_propagation(attributes[10])
print('t2', t2)

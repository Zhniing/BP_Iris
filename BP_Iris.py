from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

attributes = iris.data
num_sample = attributes.shape[0]
target = iris.target  # class label: 0 1 2
label = []
# regular the target into label(output vector)
for j in range(len(target)):
    if target[j] == 0:
        label.append(np.array([1, 0, 0]).reshape(3, 1))
    elif target[j] == 1:
        label.append(np.array([0, 1, 0]).reshape(3, 1))
    elif target[j] == 2:
        label.append(np.array([0, 0, 1]).reshape(3, 1))
# feature_names = iris.feature_names


# Size of BP

input_dim = 4  # dimension of input
output_dim = 3  # dimension of output
depth = 3  # depth of neural network
units_per_hidden = 10  # units of per hidden layer

# Parameters of BP

# w_input = np.zeros((units_per_hidden, input_dim))  # weight of input to hidden
# w_input = -1 + 2 * np.random.random((units_per_hidden, input_dim))  # weight of input to hidden
# w_hidden = np.zeros((depth-2, units_per_hidden, units_per_hidden))  # weight of hidden to hidden
# w_hidden = -1 + 2 * np.random.random((depth - 2, units_per_hidden, units_per_hidden))  # weight of hidden to hidden
# w_output = np.zeros((output_dim, units_per_hidden))  # weight of hidden to output
# w_output = -1 + 2 * np.random.random((output_dim, units_per_hidden))  # weight of hidden to output

# w = []
# w.append(-1 + 2 * np.random.random((units_per_hidden, input_dim)))
w = [-1 + 2 * np.random.random((units_per_hidden, input_dim))]
for j in range(depth - 2):
    w.append(-1 + 2 * np.random.random((units_per_hidden, units_per_hidden)))
w.append(-1 + 2 * np.random.random((output_dim, units_per_hidden)))

# b_hidden = -1 + 2 * np.random.random((depth - 1, units_per_hidden))  # bias of hidden
# b_output = -1 + 2 * np.random.random(output_dim)  # bias of output
b = []
for j in range(depth - 1):
    b.append(-1 + 2 * np.random.random((units_per_hidden, 1)))
b.append(np.random.random((output_dim, 1)))

# Record the output of every hidden layer
# output_hidden = np.zeros((depth - 1, units_per_hidden))
output = []
for j in range(depth - 1):
    output.append(np.zeros((units_per_hidden, 1)))
output.append(np.zeros((output_dim, 1)))

# Record the partial differential of the loss function for each neuron input
# differential_hidden = np.zeros((depth - 1, units_per_hidden))
# differential_output = np.zeros(output_dim)
differential = []
for j in range(depth - 1):
    differential.append(np.zeros((units_per_hidden, 1)))
differential.append(np.zeros((output_dim, 1)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


#  For a sample，run a forward propagation，return the output value
def forward_propagation(train_sample):
    z = w[0] @ train_sample + b[0]
    o = sigmoid(z)
    output[0] = o
    for i in range(1, depth - 1):
        z = w[i] @ o + b[i]
        o = sigmoid(z)
        output[i] = o
    z = w[depth - 1] @ o + b[depth - 1]
    o = softmax(z)
    output[depth - 1] = o
    return o


# Loss function: cross entropy
def backward_propagation():
    # global differential
    differential[-1] = output[-1] - label[k]
    for i in range(1, depth):
        differential[-i - 1] = w[-i].T @ differential[-i] * output[-i - 1] * (1 - output[-i - 1])


# For a sample, update the parameters
def update_parameters(x, lr):
    # lr = 0.01  # learning rate
    # declare global variable
    # global w_input
    # global b_hidden
    # global w_output
    # global b_output

    # d = differential_hidden[0]
    # d.shape = [10, 1]
    # x.shape = [1, 4]
    # w_input = w_input - lr * np.dot(differential_hidden[0].reshape(10, 1), x.reshape(1, input_dim))
    # b_hidden[0] = b_hidden[0] - lr * differential_hidden[0]
    #
    # num_hidden = w_hidden.shape[0] + 1  # the number of hidden layers
    # for i in range(num_hidden - 1):
    #     w_hidden[i] = w_hidden[i] - lr * np.dot(differential_hidden[i + 1], output_hidden[i])
    #     b_hidden[i + 1] = b_hidden[i + 1] - lr * differential_hidden[i + 1]
    #
    # differential_output.shape = [output_dim, 1]
    # o = output_hidden[num_hidden - 1]
    # o.shape = [1, 10]
    # w_output = w_output - lr * np.dot(differential_output, o)
    # b_output = b_output.reshape(output_dim, 1) - lr * differential_output

    w[0] = w[0] - lr * differential[0] @ x.T
    b[0] = b[0] - lr * differential[0]
    for i in range(1, depth):
        w[i] = w[i] - lr * differential[i] @ output[i - 1].T
        b[i] = b[i] - lr * differential[i]


def train_train(i):
    sample = attributes[i].reshape(len(attributes[i]), 1)
    forward_propagation(sample)
    backward_propagation()
    update_parameters(sample, lr)  # learning rate is 0.01


def evaluation():
    accuracy = 0
    for i in range(num_sample):
        sample = attributes[i].reshape(len(attributes[i]), 1)
        if np.argmax(forward_propagation(sample)) == target[i]:
            accuracy += 1
    accuracy /= num_sample
    print('accuracy:', str(round(accuracy * 100, 2)) + '%')


# Main
batch = 1
epoch = 200
lr = 0.001
print('Batch=%d,' % batch, 'Epoch=%d' % epoch, 'Learning_rate=%f' % lr)
print('Initial:')
evaluation()

for n in range(1000):
    for k in range(num_sample):
        train_train(k)  # input Kth sample

print('Trained:')
evaluation()

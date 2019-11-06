from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

attributes = iris.data
num_sample = attributes.shape[0]
target = iris.target  # class label: 0 1 2
label = []
# regular the target into label(vector)
for j in range(len(target)):
    if target[j] == 0:
        label.append(np.array([1, 0, 0]).reshape(3, 1))
    elif target[j] == 1:
        label.append(np.array([0, 1, 0]).reshape(3, 1))
    elif target[j] == 2:
        label.append(np.array([0, 0, 1]).reshape(3, 1))

# Size of BP

input_dim = 4  # dimension of input
output_dim = 3  # dimension of output
depth = 3  # depth of neural network
units_per_hidden = 10  # units of per hidden layer

# Parameters of BP

# w = []
# w.append(-1 + 2 * np.random.random((units_per_hidden, input_dim)))
w = [-1 + 2 * np.random.random((units_per_hidden, input_dim))]
for j in range(depth - 2):
    w.append(-1 + 2 * np.random.random((units_per_hidden, units_per_hidden)))
w.append(-1 + 2 * np.random.random((output_dim, units_per_hidden)))

b = []
for j in range(depth - 1):
    b.append(-1 + 2 * np.random.random((units_per_hidden, 1)))
b.append(np.random.random((output_dim, 1)))

# Record the output of every layer
output = []
for j in range(depth - 1):
    output.append(np.zeros((units_per_hidden, 1)))
output.append(np.zeros((output_dim, 1)))

# Record the partial differential of the loss function for each neuron input
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
    differential[-1] = output[-1] - label[k]
    for i in range(1, depth):
        differential[-i - 1] = w[-i].T @ differential[-i] * output[-i - 1] * (1 - output[-i - 1])


# For a sample, update the parameters
def update_parameters(x, lr):
    w[0] = w[0] - lr * differential[0] @ x.T
    b[0] = b[0] - lr * differential[0]
    for i in range(1, depth):
        w[i] = w[i] - lr * differential[i] @ output[i - 1].T
        b[i] = b[i] - lr * differential[i]


def train_train(i):
    sample = attributes[i].reshape(len(attributes[i]), 1)
    forward_propagation(sample)
    backward_propagation()
    update_parameters(sample, lr)


def evaluation():
    correct = 0
    for i in range(num_sample):
        if i % 2 == 0:
            continue
        sample = attributes[i].reshape(len(attributes[i]), 1)
        if np.argmax(forward_propagation(sample)) == target[i]:
            correct += 1
    accuracy = correct / (num_sample / 2)
    print('accuracy:', str(round(accuracy * 100, 2)) + '%')


# Main
epoch = 1000
lr = 0.001  # learning rate is 0.001
print('Epoch=%d' % epoch, 'Learning_rate=%f' % lr)
print('Initial:')
evaluation()

for n in range(epoch):
    for k in range(num_sample):
        if k % 2 == 0:
            train_train(k)  # input Kth sample

print('Trained:')
evaluation()

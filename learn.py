
import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm.auto import tqdm
import pickle
import gzip

# using torch for utility
from torch.utils.data import DataLoader

def downfrom(x, stop=-1, step=-1):
    return range(x, stop, step)

def onehot(v, size):
    vec = np.zeros(10)
    vec[v] = 1.0
    return vec

class Network():
    def __init__(self, shape, lr=1e-1):
        self.lr = lr
        self.params = Network.new_params(shape)
        self.inputs = Network.new_activations(shape, bias=False)
        self.activations = Network.new_activations(shape)

        self.dparams = Network.new_params(shape)
        self.dactivations = Network.new_activations(shape, bias=False)
    
    def forward_sample(self, data):
        bias = 1.0
        # "0th" layer activation is the data
        self.activations[0][:-1] = data
        self.activations[0][-1] = bias
        for L in range(1, len(self.activations)):
            # calculate inputs from previous layer activations
            self.inputs[L][:] = self.params[L].dot(self.activations[L-1])
            # calculate activations from inputs
            self.activations[L][:-1] = sigmoid(self.inputs[L])
            # bias is implicit 
            self.activations[L][-1] = bias
        return self.activations[-1][:-1]

    def backprop(self, target):
        self.dactivations[-1] = dloss(self.activations[-1][:-1], target)
        for L in downfrom(len(self.params)-1, 0):
            # gradient of inputs to layer
            dinputs = self.dactivations[L] * dsigmoid(self.inputs[L])
            # gradient of parameters of layer
            dparams = np.outer(dinputs, self.activations[L-1])
            # accumulate gradients
            self.dparams[L] -= dparams
            # gradient of previous layer
            self.dactivations[L-1] = self.params[L].transpose().dot(dinputs)[:-1]
    
    def update(self):
        """Update parameters based on gradient"""
        for param, grad in zip(self.params[1:], self.dparams[1:]):
            param += self.lr*grad
            grad.fill(0)

    def forward(self, batch, grad=True):
        images, labels = batch
        images = np.array(images).reshape(images.shape[0], -1)/255.0
        labels = np.array(labels)

        avg_loss = 0
        avg_correct = 0
        for batchn, (x, y) in enumerate(zip(images, labels)):
            target = onehot(y, 10)
            output = self.forward_sample(x)
            correct = (np.argmax(output) == y)
            avg_correct += correct
            avg_loss += loss(output, target)
            if grad:
                self.backprop(target)
                self.update()
        avg_correct = float(avg_correct) / len(labels)
        avg_loss /= len(batch)
        return (avg_loss, avg_correct)


    @staticmethod
    def new_params(sizes):
        net = [None]
        for a, b in zip(sizes, sizes[1:]):
            # Linear layer size a->b with bias
            net.append(np.random.normal(size=(b, a+1), scale=1/math.sqrt(a)))
        return net

    @staticmethod
    def new_activations(sizes, bias=True):
        net = []
        for size in sizes:
            net.append(np.zeros(size+bias))
        return net

def loss(output, target):
    """loss function"""
    return np.mean((output-target)**2)

def dloss(output, target):
    """derivative of loss"""
    return 2*(output-target)

def sigmoid(x):
    """vectorized sigmoid function"""
    return 1/(1 + np.exp(-x))

def dsigmoid(x):
    """vectorized derivative of sigmoid"""
    o = sigmoid(x)
    return o*(1-o)

def main():
    with gzip.open("data/mnist.pickle.gz", "rb") as f:
        train_set, test_set = pickle.load(f)
    train_data, train_labels = train_set
    test_data, test_labels = test_set
    batch_size = 64

    network_shape = [28*28, 16, 16, 10]
    net = Network(network_shape)
    epochs = 3
    for epoch in range(epochs):
        batches = [
            (train_data[i:i+batch_size], train_labels[i:i+batch_size])
            for i in range(0, len(train_data), batch_size)
        ]
        avg_loss = 0
        avg_correct = 0
        for batch in tqdm(batches):
            loss, correct = net.forward(batch, grad=True)
            avg_loss += loss
            avg_correct += correct
        print(f"EPOCH {epoch} loss {avg_loss/len(batches):.3f} correct {avg_correct/len(batches)*100:.1f}% lr {net.lr}")
        net.lr *= 0.1
    
    avg_correct = 0
    test_batches = [
        (test_data[i:i+batch_size], test_labels[i:i+batch_size])
        for i in range(0, len(test_data), batch_size)
    ]
    for batch in test_batches:
        loss, correct = net.forward(batch, grad=False)
        avg_correct += correct
    print(f"TEST correct {avg_correct/len(test_batches)*100:.1f}%")

    
if __name__ == "__main__":
    main()
from mnist import MNIST
import numpy as np
from layers import *


class Model(object):

    def __init__(self, input_dim=28*28, first_hidden_dim=500, second_hidden_dim=500, num_classes=10,
                 weight_scale=1e-3, reg=0.0):

        self.params = {}
        self.reg = reg
        self.D = input_dim
        self.H1 = first_hidden_dim
        self.H2 = second_hidden_dim
        self.C = num_classes

        self.params['W1'] = np.random.normal(0.0, weight_scale, (self.D, self.H1))
        self.params['b1'] = np.zeros(self.H1)
        self.params['W2'] = np.random.normal(0.0, weight_scale, (self.H1, self.H2))
        self.params['b2'] = np.zeros(self.H2)
        self.params['W3'] = np.random.normal(0.0, weight_scale, (self.H2, self.C))
        self.params['b3'] = np.zeros(self.C)

    # A model object implements a loss function which accepts a batch of data and its corresponding labels,
    # and returns a numerical loss along with a dictionary of gradients for that pass.
    def loss(self, X, y=None):

        H1_activations, H1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        H2_activations, H2_cache = affine_relu_forward(H1_activations, self.params['W2'], self.params['b2'])
        scores, score_cache = affine_forward(H2_activations, self.params['W3'], self.params['b3'])

        if y is None:
            return scores

        data_loss, data_loss_dx = softmax_loss(scores, y)
        loss = data_loss + (0.5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2) +
                            np.sum(self.params['W3']**2)))

        grads = {}

        H2_dx, grads['W3'], grads['b3'] = affine_backward(data_loss_dx, score_cache)
        H1_dx, grads['W2'], grads['b2'] = affine_relu_backward(H2_dx, H2_cache)
        X_dx, grads['W1'], grads['b1'] = affine_relu_backward(H1_dx, H1_cache)
        grads['W3'] += self.reg * self.params['W3']
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']

        return loss, grads


class Solver(object):

    def __init__(self, model, data, **kwargs):
        self.model = model
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_test = data['x_test']
        self.y_test = data['y_test']
        self.batch_size = 100
        self.num_epochs = 5
        self.print_every = 50
        self.loss_history = []
        self.cache = {}
        for p in self.model.params:
            self.cache[p] = {}

    def step(self):

        N = self.x_train.shape[0]
        batch_mask = np.random.choice(N, self.batch_size)
        x_batch = self.x_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        loss, grads = self.model.loss(x_batch, y_batch)
        self.loss_history.append(loss)

        for p, w in self.model.params.items():
            dw = grads[p]
            # if cache then else none
            cache = self.cache[p]
            next_w, next_cache = sgd_momentum(w, dw, cache)
            self.cache[p] = next_cache
            self.model.params[p] = next_w

    def check_accuracy(self, x, y, num_samples=None, batch_size=100):

        N = x.shape[0]
        if num_samples is not None and num_samples < N:
            mask = np.random.choice(N, num_samples)
            x_batch = x[mask]
            y_batch = y[mask]

        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(int(num_batches)):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(x[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc


    def train(self):

        num_train = self.x_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = int(np.floor(self.num_epochs * iterations_per_epoch))

        for i in range(num_iterations):
            self.step()

            if i % self.print_every == 0:
                print('Iteration {} / {} loss: {}'.format(i, num_iterations, self.loss_history[-1]))

            if (i + 1) % iterations_per_epoch == 0:
                accuracy = self.check_accuracy(self.x_test, self.y_test, num_samples=2000)
                print('Epoch {} accuracy: {}'.format(((i + 1) / iterations_per_epoch), accuracy))





mnist_data = MNIST('/Users/Harry/Documents/MNIST-Classifier/python-mnist/data')
training_images, training_labels = mnist_data.load_training()
test_images, test_labels = mnist_data.load_testing()
data = {'x_train': np.array(training_images), 'y_train': np.array(training_labels),
        'x_test': np.array(test_images), 'y_test': np.array(test_labels)}

digit_net = Model()
digit_solver = Solver(digit_net, data)
digit_solver.train()

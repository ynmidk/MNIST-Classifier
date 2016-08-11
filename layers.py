import numpy as np

# All forward layers accept; a batch of data X, a np array of weights, a np array of baises.
# All forward layers return a np array of activations, a cache of data for the back pass.

# X has shape (N, D), where N is number of examples in the mini-batch
# and D is the number of dimensions in each example.
# W has shape (D, H). b has shape H.
# Activations of layer have shape (N, H)
# Upstream gradient has shape (N, H)
# you multiply each component by the upstream gradient to get the real effect that parameter had on the final loss,
# not just the local activation.

def affine_forward(X, W, b):

    activations = np.dot(X, W) + b
    cache = (X, W, b)

    return activations, cache

def affine_backward(upstream_gradient, cache):

    X, W, b = cache
    db = np.sum(upstream_gradient, axis=0)
    dW = np.dot(X.T, upstream_gradient)
    dX = np.dot(upstream_gradient, W.T)

    return dX, dW, db

def relu_forward(X):

    activations = np.maximum(0.0, X)
    cache = X

    return activations, cache

def relu_backward(upstream_gradient, cache):

    dx = upstream_gradient
    dx[cache <= 0] = 0

    return dx

def affine_relu_forward(X, W, b):

    activations_affine, cache_affine = affine_forward(X, W, b)
    activations_relu, cache_relu = relu_forward(activations_affine)

    return activations_relu, (cache_affine, cache_relu)

def affine_relu_backward(upstream_gradient, cache):

    cache_affine, cache_relu = cache
    drelu = relu_backward(upstream_gradient, cache_relu)
    dX, dW, db = affine_backward(drelu, cache_affine)

    return dX, dW, db

def softmax_loss(x, y):

  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx


def sgd_momentum(w, dw, cache=None):

    if cache is None: cache = {}
    cache.setdefault('momentum', 0.9)
    cache.setdefault('learning_rate', 1e-3)
    v = cache.get('velocity', np.zeros_like(w))

    v = ((cache['momentum'] * v) - (cache['learning_rate'] * dw))
    next_w = w + v
    cache['velocity'] = v

    return next_w, cache

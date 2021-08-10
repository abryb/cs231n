from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    for i in range(num_train):
        y_true = y[i]
        x = X[i]
        # forward
        H = X[i].dot(W)
        A = np.exp(H)/sum(np.exp(H))
        J = -np.log(A[y_true]) 
        loss += J

        # backward
        dJ_dH = A.copy()
        dJ_dH[y_true] -= 1 # (C,)
        dH_dW = np.zeros_like(dW) + x[:,None] # (D,C)
        dJ_dW = dJ_dH * dH_dW       
        dW += dJ_dW
        
     
    loss /= num_train
    dW /= num_train
    
    loss += reg * np.sum(W*W)
    dW += reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # forward
    num_train = X.shape[0]
    H = X @ W # (N,C)
    H = H - np.max(H, axis=1, keepdims=True)
    A = np.exp(H)
    A /= A.sum(axis=1)[:,None] # (N,C)
    J = np.sum(-np.log(A[np.arange(num_train),y]))
    loss = (J / num_train) + reg * np.sum(W*W)
    # backward
    dJ_dH = A.copy() # (N,C)
    dJ_dH[np.arange(num_train),y] -= 1 # (N,C)    
    dW = X.T.dot(dJ_dH)
    dW /= num_train
    dW +=  reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

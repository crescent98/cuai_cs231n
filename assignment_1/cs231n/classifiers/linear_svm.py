import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) # scores = X dot W
    correct_class_score = scores[y[i]] # correct_class_score는 scores[training labels]
    for j in xrange(num_classes): # class를 돌면서
      if j == y[i]: # class가 해당 training label일 경우 pass
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1, svm loss식
      if margin > 0:        #margin이 0보다 작을 경우는 0이라서 더해줄 필요 x, margin이 0보다 클 경우 loss에 margin을 더해줌
        loss += margin      # dmax(0,sj-sy[i]+1)/dW = dW,  s= XW => ds/dW = X
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W  

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # dW: loss의 미분                                                           #
  # Rather that first computing the loss and then computing the derivative,   #
  # loss를 먼저 계산하고 미분을 계산하는 것 대신에, 동시에 loss, 미분 계산이  #
  # 더 간단할 수 있음. 그래서 위에 gradient를 계산하는 코드가 수정될 수 있음  # 
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  # loss에 대한 result를 저장하는 SVM loss의 vectorized version을 구현        #
  #############################################################################
  
  num_train = X.shape[0]
    
  scores = np.dot(X, W) #scores = (N,C), y= (N,)
  correct_class_scores = np.choose(y, scores.T).reshape(-1, 1)
  
  margin = np.maximum(scores-correct_class_scores+1, 0.0)
  margin[np.arange(num_train), y] = 0.0
  
  loss = np.sum(margin)/ num_train
  loss += reg * np.sum(W*W)
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin[margin > 0] = 1 
  margin[np.arange(y.shape[0]), y] -= np.sum(margin, axis=1)
    
  dW = np.dot(X.T, margin)
  
  dW /= num_train
  dW += reg * 2 * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

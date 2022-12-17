from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange

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
    # TODO: 使用显式循环计算softmax损失及其梯度。
    # 将损失和梯度分别保存在loss和dW中。
    # 如果你不小心，很容易遇到数值不稳定的情况。 
    # 不要忘了正则化！                                                           
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    
    for i in range(num_train):
        scores = X[i].dot(W)
        scores = np.exp(scores).reshape(1, -1)
        score_sum = np.sum(scores)
        
        loss -= np.log(scores[0,y[i]] / score_sum)
        
        
        # 原始式： -(score_sum / scores[y[i]]) * (scores[y[i]]*score_sum*X[i] - scores[y[i]]*scores[y[i]]*X[i]) / (score_sum**2)
        #     dW[:,y[i]] += -(score_sum - scores[0,y[i]]) / score_sum * X[i].T
        # 原始式：-(score_sum / scores[y[i]]) * (-scores[y[i]] / (score_sum ** 2)) * X[i].T.dot(scores)
        #     dW += 1 / score_sum * X[i].T.reshape(-1, 1).dot(scores)
        # y[i]列的dW已经在最开始统一计算，因此上面一步额外计算的y[i]列梯度要减去
        #     dW[:,y[i]] -= scores[0,y[i]] / score_sum * X[i].T
        # 抵消dW[:,y[i]]的两项，得：
        
        dW[:,y[i]] -= X[i].T
        dW += 1 / score_sum * X[i].T.reshape(-1, 1).dot(scores)
        
    loss /= num_train
    dW /= num_train # dW计算
    
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W # dW计算
        
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
    # TODO: 不使用显式循环计算softmax损失及其梯度。
    # 将损失和梯度分别保存在loss和dW中。
    # 如果你不小心，很容易遇到数值不稳定的情况。 
    # 不要忘了正则化！
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    
    scores = X.dot(W)
    scores = np.exp(scores)
    scores_sum = np.sum(scores, axis=1).reshape(-1, 1)
    p_scores = scores / scores_sum
    p_scores[range(num_train), y] -= 1
    correct_class_scores = scores[range(num_train), y].reshape(-1, 1)
    
    loss -= np.sum(np.log(correct_class_scores / scores_sum))
    dW += X.T.dot(p_scores)
        
    loss /= num_train
    dW /= num_train # dW计算
    
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W # dW计算

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

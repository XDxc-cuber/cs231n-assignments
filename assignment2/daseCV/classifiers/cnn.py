from builtins import object
import numpy as np

from daseCV.layers import *
from daseCV.fast_layers import *
from daseCV.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: 初始化三层卷积网络的权重和偏差。
        #     权重应用以0.0为中心的高斯初始化，标准差等于weight_scale；
        #     偏差应初始化为零。所有权重和偏差应存储在字典self.params中。
        #     使用键“ W1”和“ b1”存储卷积层的权重和偏差；使用键“ W2”和“ b2”
        #     表示隐藏仿射层的权重和偏差，并使用键“ W3”和“ b3”表示输出仿射层的
        #     权重和偏差。重要说明：对于本次作业，您可以假设第一个卷积层的padding和
        #     stride已经设置了，这样**输入的width和height就保留了**。看一下loss()
        #     函数的前部分它是如何做的。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        num_channels, H, W = input_dim
        stride, pad = 1, (filter_size-1) // 2
        pool_h, pool_w, pool_stride = 2, 2, 2
        
        HH = 1 + (H + 2 * pad - filter_size) // stride
        WW = 1 + (W + 2 * pad - filter_size) // stride
        H_pool, W_pool = 1 + (HH - pool_h) // pool_stride, 1 + (WW - pool_h) // pool_stride
        layer2_input_dim = num_filters * H_pool * W_pool
        
        self.params['W1'] = np.random.normal(loc=0., scale=weight_scale, size=(num_filters, num_channels, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(loc=0., scale=weight_scale, size=(layer2_input_dim, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(loc=0., scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)
    
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: 实现三层卷积网络的正向传播，计算X的每类的分数并将其存储在scores变量中。
        #请注意，您可以在实现中使用daseCV/fast_layers.py和daseCV/layer_utils.py中定义的功能（已导入）。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
#         conv_shape = conv_out.shape
#         conv_out = conv_out.reshape(conv_out.shape[0], W2.shape[0])
        layer2_out, layer2_cache = affine_relu_forward(conv_out, W2, b2)
        scores, scores_cache = affine_forward(layer2_out, W3, b3)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: 完成三层卷积网络的反向传播，将损失和梯度存储在变量loss和grads中。
        # 使用softmax计算损失，并使用grads[k]保存self.params[k]的梯度。不要忘记增加L2正则化！
        #                                              
        # NOTE: 为确保您的实现与我们的实现相同并通过自动化测试，请确保您的L2正则化系数为0.5，
        # 以简化梯度的表达式。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dout = softmax_loss(scores, y)
        reg_sum = np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2)
        reg_sum *= 0.5 * self.reg
        loss += reg_sum
        
        dout, grads['W3'], grads['b3'] = affine_backward(dout, scores_cache)
        dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, layer2_cache)
        dout, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, conv_cache)
        
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

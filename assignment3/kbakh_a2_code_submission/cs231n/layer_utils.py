from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer.
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def generic_forward(x, w, b, gamma=None, beta=None, bn_param=None, dropout_param=None, last=False):
    result_caches = [None] * 4
    
    transformed, fc_result = affine_forward(x, w, b)
    
    if not last:
        if bn_param is not None:
            if 'mode' in bn_param:
               
                if gamma is None:
                    gamma = np.ones(transformed.shape[1])
                if beta is None:
                    beta = np.zeros(transformed.shape[1])
                transformed, result_caches[0] = batchnorm_forward(transformed, gamma, beta, bn_param)
            else:
               
                if gamma is None:
                    gamma = np.ones(transformed.shape[1])
                if beta is None:
                    beta = np.zeros(transformed.shape[1])
                transformed, result_caches[1] = layernorm_forward(transformed, gamma, beta, bn_param)
                
        transformed, result_caches[2] = relu_forward(transformed)
        
        if dropout_param:
            transformed, result_caches[3] = dropout_forward(transformed, dropout_param)
    
    all_cache = (fc_result, *result_caches)
    
    return transformed, all_cache


def generic_backward(dout, cache):
    grad_norm_params = [None, None]
    
    all_fc_cache, *norm_activation_caches = cache
    
    if norm_activation_caches[3] is not None:
        dout = dropout_backward(dout, norm_activation_caches[3])
    
    if norm_activation_caches[2] is not None:
        dout = relu_backward(dout, norm_activation_caches[2])
    
    if norm_activation_caches[0] is not None:
        dout, grad_norm_params[0], grad_norm_params[1] = batchnorm_backward_alt(dout, norm_activation_caches[0])
    elif norm_activation_caches[1] is not None:
        dout, grad_norm_params[0], grad_norm_params[1] = layernorm_backward(dout, norm_activation_caches[1])
    
    input_grad, weight_grad, bias_grad = affine_backward(dout, all_fc_cache)
    
    return input_grad, weight_grad, bias_grad, grad_norm_params[0], grad_norm_params[1]

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def conv_relu_forward(x, w, b, conv_param):
    """A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for the conv-bn-relu convenience layer.
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """Backward pass for the conv-relu-pool convenience layer.
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

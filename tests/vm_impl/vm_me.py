# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""VM implementations based on numpy."""

import numpy as np
from mindspore._checkparam import Rel
from mindspore._checkparam import ParamValidator as validator


def avg_pooling(x, pool_h, pool_w, stride):
    """
    Applies average pooling over an input array.

    Args:
        x (numpy.ndarray): The input array to be average pooled.
        pool_h (int): Height of the pooling window.
        pool_w (int): Width of the pooling window.
        stride (int): The stride of the sliding window.

    Returns:
        numpy.ndarray, an output array after applying average pooling on input array.
    """
    validator.check_integer("stride", stride, 0, Rel.GT)
    num, channel, height, width = x.shape
    out_h = (height - pool_h)//stride + 1
    out_w = (width - pool_w)//stride + 1

    col = im2col(x, pool_h, pool_w, stride)
    col = col.reshape(-1, pool_h*pool_w)

    out = np.mean(col, axis=1)
    out = out.reshape((num, out_h, out_w, channel)).transpose(0, 3, 1, 2)

    return out


def avg_pool_grad(dout, origin_shape, pool_h, pool_w, stride):
    """
    Gets grad of average pooling.

    Args:
        x (numpy.ndarray): The input array to be average pooled.
        dout (numpy.ndarray): The  grad of pre-layer.
        pool_h (int): Height of the pooling window.
        pool_w (int): Width of the pooling window.
        stride (int): The stride of the sliding window.

    Returns:
        numpy.ndarray, grad of avgerage pooling.
    """
    # pylint: disable=unused-argument
    _, _, height, width = dout.shape
    dx = np.zeros(origin_shape)
    for i in range(height):
        for j in range(width):
            dx[:, :, i:(i+pool_h), j:(j+pool_w)] += np.ones((pool_h, pool_w))
    return dx


def _batch_norm(x, scale, shift, running_mean=None, running_var=None,
                eps=1e-05, momentum=0.1, is_training=True):
    """Batch normalization over an array."""
    _, c_h_w = x.shape
    # Handle running_mean and running_var are not None
    # if running_mean is None:
    #     running_mean = np.zeros(c_h_w)
    #     running_var = np.zeros(c_h_w)
    running_mean = np.zeros(c_h_w)
    running_var = np.zeros(c_h_w)
    if np.ndim(scale) > 0:
        scale = scale.mean()
    if np.ndim(shift) > 0:
        shift = shift.mean()

    if is_training:
        x_mean = np.mean(x, axis=0)
        x_var = np.var(x, axis=0)

        # Normalization followed by Affine transformation
        x_norm = (x - x_mean)/np.sqrt(x_var + eps)

        # Estimate running average of mean and variance to use at test time
        running_mean = momentum * running_mean + (1 - momentum) * x_mean
        running_var = momentum * running_var + (1 - momentum) * x_var
    else:
        # normalize using running average
        x_norm = (x - running_mean)/np.sqrt(running_var + eps)
        x_mean = running_mean
        x_var = running_var

    out = scale * x_norm + shift

    return out, x_mean, x_var, running_mean, running_var


def batch_norm(x, scale=1, shift=0, mean=None, variance=None,
               eps=1e-05, momentum=0.1, is_training=True):
    """Batch normalization over an array."""
    input_shape = x.shape
    if x.ndim != 2:
        batch_num = x.shape[0]
        x = x.reshape(batch_num, -1)

    out, _, _, running_mean, running_var = _batch_norm(x, scale, shift, mean, variance, \
                                                       eps, momentum, is_training)

    return out.reshape(*input_shape), np.array(scale), np.array(shift), running_mean, running_var


def _batch_norm_grad(dout, x, scale, save_mean, save_inv_variance, \
                     eps=1e-05, momentum=0.1, is_training=True):
    """Batch normalization over an array."""
    if x.ndim != 2:
        batch_num = x.shape[0]
        x = x.reshape(batch_num, -1)
    if np.ndim(scale) > 0:
        scale = scale.mean()
    x_norm, x_mean, x_var, _, _ = _batch_norm(x, scale, shift=0, running_mean=save_mean, \
                                              running_var=save_inv_variance, \
                                              eps=eps, momentum=momentum, is_training=is_training)
    batch_size = x.shape[0]
    dx_norm = scale * dout
    dvar = np.sum(dx_norm*(x - x_mean)*((x_var + eps)**(-3.0/2))*(-1.0/2), axis=0)
    dmean = np.sum(dx_norm*(-1.0/np.sqrt(x_var + eps)), axis=0) \
                            + dvar*(np.sum(-2*(x - x_mean), axis=0)*(1.0/batch_size))
    dx = dx_norm*(1.0/np.sqrt(x_var + eps)) + dvar*(2.0*(x - x_mean)/batch_size) + dmean*(1.0/batch_size)
    dgamma = np.sum(dout*x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    return dx, dgamma, dbeta


def batch_norm_grad(dy, x, scale, save_mean, save_inv_variance):
    """Batch normalization over an array."""
    if dy.ndim != 2:
        batch_size = dy.shape[0]
        dy = dy.reshape(batch_size, -1)

    dx, dgamma, dbeta = _batch_norm_grad(dy, x, scale, save_mean, save_inv_variance)
    input_shape = x.shape
    dx = dx.reshape(*input_shape)
    return dx, dgamma, dbeta


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """Rearranges a row vector to an image."""
    if isinstance(stride, int):
        stride_h = stride
        stride_w = stride
    elif isinstance(stride, tuple) and len(stride) == 2:
        stride_h = stride[0]
        stride_w = stride[1]
    elif isinstance(stride, tuple) and len(stride) == 3:
        stride_h = stride[2]
        stride_w = stride[3]
    else:
        raise ValueError(f"The \'stride\' should be an int number or "
                         f"a tuple of two or four int numbers, but got {stride}")

    batch_num, channel, height, width = input_shape
    out_h = (height + 2*pad - filter_h)//stride_h + 1
    out_w = (width + 2*pad - filter_w)//stride_w + 1
    col = col.reshape(batch_num, out_h, out_w, channel, filter_h, filter_w) \
             .transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((batch_num,
                    channel,
                    height + 2*pad + stride_h - 1,
                    width + 2*pad + stride_w - 1)) \
            .astype(col.dtype)
    for y in range(filter_h):
        y_max = y + stride_h*out_h
        for x in range(filter_w):
            x_max = x + stride_h*out_w
            img[:, :, y:y_max:stride_h, x:x_max:stride_h] += col[:, :, y, x, :, :]

    return img[:, :, pad:height + pad, pad:width + pad]


def convolve(x, w, b=None, pad_mode="valid"):
    """
    Gets the discrete, linear convolution of two one-dimensional sequences.

    Args:
        x (numpy.ndarray): One-dimensional input array.
        w (numpy.ndarray): One-dimensional input array.
        b (numpy.ndarray): One-dimensional input array. Default: None.
        pad_mode (str): Padding mode which can be: "full" means returns the
                  convolution at each point of overlap, with an output shape
                  of (N+M-1,); "same" means returns output of length max(M, N);
                  Amd "valid" means returns output of length max(M, N) - min(M, N)
                  + 1. Default: "valid".

    Returns:
        numpy.ndarray, discrete, linear convolution of x and w, then plus b.
    """
    if pad_mode not in {"same", "valid"}:
        pad_mode = "full"
    y = np.convolve(x, w, pad_mode)
    if b:
        y += b
    return y


def conv2d(x, weight, bias=None, stride=1, pad=0,
           dilation=1, groups=1, padding_mode='zeros'):
    """Convolution 2D."""
    # pylint: disable=unused-argument
    validator.check_type('stride', stride, (int, tuple))
    if isinstance(stride, int):
        stride = (stride, stride)
    elif len(stride) == 4:
        stride = (stride[2], stride[3])
    if len(stride) != 2 or (not isinstance(stride[0], int)) or \
                           (not isinstance(stride[1], int)) or \
                           stride[0] < 1 or stride[1] < 1:
        raise ValueError(f"The \'stride\' of \'conv2d\' should be an positive int number or "
                         f"a tuple of two positive int numbers, but got {stride}")
    stride_h = stride[0]
    stride_w = stride[1]
    validator.check_type('dilation', dilation, (int, tuple))
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    elif len(dilation) == 4:
        dilation = (dilation[2], dilation[3])
    if len(dilation) != 2 or (not isinstance(dilation[0], int)) or \
                           (not isinstance(dilation[1], int)) or \
                           dilation[0] < 1 or dilation[1] < 1:
        raise ValueError(f"The \'dilation\' of \'conv2d\' should be an positive int number or "
                         f"a tuple of two positive int numbers, but got {dilation}")
    dilation_h = dilation[0]
    dilation_w = dilation[1]

    batch_num, _, x_h, x_w = x.shape
    filter_num, _, filter_h, filter_w = weight.shape
    out_h = 1 + int((x_h + 2 * pad - filter_h - (filter_h - 1) * (dilation_h - 1)) / stride_h)
    out_w = 1 + int((x_w + 2 * pad - filter_w - (filter_w - 1) * (dilation_w - 1)) / stride_w)
    col = im2col(x, filter_h, filter_w, stride, pad, dilation)
    col_w = np.reshape(weight, (filter_num, -1)).T
    out = np.dot(col, col_w)
    out = out.reshape(batch_num, out_h, out_w, -1).transpose(0, 3, 1, 2)
    if bias is not None:
        out += bias
    return out


def conv2d_backprop_filter(dout, x, w_size, stride=1, pad=0):
    """Backpropagation filter for conv2d."""
    filter_num, channel, filter_height, filter_width = w_size
    dout = dout.transpose(0, 2, 3, 1).reshape(-1, filter_num)
    col = im2col(x, filter_height, filter_width, stride, pad)
    dw = np.dot(col.T, dout)
    dw = dw.transpose(1, 0).reshape(filter_num, channel, filter_height, filter_width)
    return dw


def conv2d_backprop_input(dout, x_size, weight, stride=1, pad=0):
    """Backpropagation input for conv2d."""
    filter_num, _, filter_h, filter_w = weight.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(-1, filter_num)
    col_w = weight.reshape(filter_num, -1).T
    dcol = np.dot(dout, col_w.T)
    dx = col2im(dcol, x_size, filter_h, filter_w, stride, pad)
    return dx


def flatten(x):
    """
    Flattens an array to one dimension.

    Args:
        x (numpy.ndarray): An array to be flattened.

    Returns:
        numpy.ndarray, a flattened array in one dimension.
    """
    return x.flatten()


def flatten2(x):
    """
    Flattens an array to one dimension by reshape.

    Args:
        x (numpy.ndarray): An array to be flattened.

    Returns:
        numpy.ndarray, a flattened array in one dimension.
    """
    return x.reshape(1, -1)


def flatten_batch(x):
    """
    Flattens a batch of arrays to one dimension.

    Args:
        x (numpy.ndarray): A batch of arrays to be flattened.

    Returns:
        numpy.ndarray, a flattened one dimension array.
    """
    return x.reshape(x.shape[0], -1)


def flatten_grad(dout, x):
    """Grad of flatten."""
    dout = np.reshape(dout, x)
    return dout


def im2col(img, filter_h, filter_w, stride=1, pad=0, dilation=1):
    """Rearranges an image to row vector."""
    if isinstance(stride, int):
        stride_h = stride
        stride_w = stride
    elif isinstance(stride, tuple) and len(stride) == 2:
        stride_h = stride[0]
        stride_w = stride[1]
    elif isinstance(stride, tuple) and len(stride) == 3:
        stride_h = stride[2]
        stride_w = stride[3]
    else:
        raise ValueError(f"The \'stride\' should be an int number or "
                         f"a tuple of two or four int numbers, but got {stride}")
    if isinstance(dilation, int):
        dilation_h = dilation
        dilation_w = dilation
    elif isinstance(dilation, tuple) and len(dilation) == 2:
        dilation_h = dilation[0]
        dilation_w = dilation[1]
    elif isinstance(dilation, tuple) and len(dilation) == 3:
        dilation_h = dilation[2]
        dilation_w = dilation[3]
    else:
        raise ValueError(f"The \'dilation\' should be an int number or "
                         f"a tuple of two or four int numbers, but got {dilation}")

    batch_num, channel, height, width = img.shape
    out_h = (height + 2*pad - filter_h- (filter_h - 1) * (dilation_h - 1))//stride_h + 1
    out_w = (width + 2*pad - filter_w- (filter_w - 1) * (dilation_w - 1))//stride_w + 1

    img = np.pad(img, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((batch_num, channel, filter_h, filter_w, out_h, out_w)).astype(img.dtype)

    for y in range(filter_h):
        y_max = y + stride_h*out_h
        for x in range(filter_w):
            x_max = x + stride_h*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride_h, x:x_max:stride_h]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_num*out_h*out_w, -1)
    return col


def matmul(x, w, b=None):
    """
    Dot product of array x and w, then plus array b if b is not None.

    Args:
        x (numpy.ndarray): Represents the input array.
        w (numpy.ndarray): Represents weights array.
        b (numpy.ndarray): Represents bias array which has the same shape as x. Default: None.

    Returns:
        numpy.ndarray, the result of (x*w + b).
    """
    y = np.dot(x, w)
    if b:
        y += b
    return y


def max_pooling(x, pool_h, pool_w, stride):
    """Max pooling."""
    validator.check_integer("stride", stride, 0, Rel.GT)
    num, channel, height, width = x.shape
    out_h = (height - pool_h)//stride + 1
    out_w = (width - pool_w)//stride + 1

    col = im2col(x, pool_h, pool_w, stride)
    col = col.reshape(-1, pool_h*pool_w)

    out = np.max(col, axis=1)
    out = out.reshape((num, out_h, out_w, channel)).transpose(0, 3, 1, 2)

    return out


def max_pool_grad(x, dout, pool_h, pool_w, stride):
    """Grad of max pooling."""
    dout = dout.transpose(0, 2, 3, 1)
    pool_size = pool_h * pool_w
    dmax = np.zeros((dout.size, pool_size))
    col = im2col(x, pool_h, pool_w, stride)
    col = col.reshape(-1, pool_h*pool_w)
    arg_max = np.argmax(col, axis=1)
    dmax[np.arange(arg_max.size), arg_max.flatten()] = dout.flatten()
    dmax = dmax.reshape(dout.shape + (pool_size,))
    dcol = dmax.reshape(dmax.shape[0]*dmax.shape[1]*dmax.shape[2], -1)
    dx = col2im(dcol, x.shape, pool_h, pool_w, stride)
    return dx


def max_pool_grad_with_argmax(x, dout, arg_max, pool_h, pool_w, stride):
    """Grad of max pooling with argmax."""
    dout = dout.transpose(0, 2, 3, 1)
    pool_size = pool_h * pool_w
    dmax = np.zeros((dout.size, pool_size))
    dmax[np.arange(arg_max.size), arg_max.flatten()] = dout.flatten()
    dmax = dmax.reshape(dout.shape + (pool_size,))
    dcol = dmax.reshape(dmax.shape[0]*dmax.shape[1]*dmax.shape[2], -1)
    dx = col2im(dcol, x.shape, pool_h, pool_w, stride)
    return dx


def max_pool_with_argmax(x, pool_h, pool_w, stride):
    """Max pooling with argmax."""
    validator.check_integer("stride", stride, 0, Rel.GT)
    num, channel, height, width = x.shape
    out_h = (height - pool_h)//stride + 1
    out_w = (width - pool_w)//stride + 1
    col = im2col(x, pool_h, pool_w, stride)
    col = col.reshape(-1, pool_h*pool_w)
    out = np.max(col, axis=1)
    out_argmax = np.argmax(col, axis=1)
    out = out.reshape((num, out_h, out_w, channel)).transpose(0, 3, 1, 2)
    out_argmax = out_argmax.reshape((num, out_h, out_w, channel)).transpose(0, 3, 1, 2)
    return out, out_argmax


def relu(x):
    """
    Rectified linear unit.

    Args:
        x (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray, the array applied relu.
    """
    return x * (x > 0)


def relu_grad(y):
    """
    Grad of relu.

    Args:
        y (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray, the array applied grad of relu.
    """
    y[y <= 0] = 0
    y[y > 0] = 1
    return y


def sigmoid(x):
    """
    Sigmoid activation function.

    Args:
        x (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray, the array applied sigmoid.
    """
    return 1 / (1 + np.exp(x * -1))


def tanh(x):
    """
    Computes hyperbolic tangent element-wise.

    Args:
        x (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray, the array applied tanh.
    """
    a = np.exp(x) - np.exp(x * -1)
    b = np.exp(x) + np.exp(x * -1)
    return a / b


def softmax(x, axis=None):
    """
    Softmax function which is `softmax(x) = np.exp(x)/sum(np.exp(x))`.

    Args:
        x (numpy.ndarray): Input array.
        axis (Union[int, tuple[int]]): Axis to compute values along. Default: None.

    Returns:
        numpy.ndarray, has the same shape as x.
    """
    from scipy.special import softmax as scipy_softmax
    return scipy_softmax(x, axis)


def softmax_cross_entropy_with_logits(logits, labels):
    sample_num = labels.shape[0]
    prob = softmax(logits)
    log_likelihood = -np.log(prob[range(sample_num)]) * labels
    #loss = np.sum(log_likelihood)
    loss = log_likelihood

    dx = prob.copy()
    dx[range(sample_num)] -= labels
    return loss, dx


def shape(x):
    """
    Gets the array's dimensions.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        tuple, the shape/dimensions of the input array.
    """
    return np.array(np.shape(x))


def expand_dims(x, axis):
    """
    Expands the shape of an array.

    Args:
        x (numpy.ndarray): Input array.
        axis (int): Position in the expanded axes where the new axis is placed.

    Returns:
        numpy.ndarray, view of input array with the number of dimensions increased by one.
    """
    return np.expand_dims(x, axis)


def squeeze(x, axis):
    """
    Removes single-dimensional entries from the shape of an array.

    Args:
        x (numpy.ndarray): Input array.
        axis (Union[int, tuple[int]]): Selected subset of the single-dimensional entries in the shape.

    Returns:
        numpy.ndarray, the input numpy.ndarray, but with all or a subset of the dimensions of length
        1 removed.
    """
    return np.squeeze(x, tuple(axis))


def reshape(x, shp):
    """
    Applies a new shape to an array without changing its data.

    Args:
        x (numpy.ndarray): Input array.
        shp (tuple[int]): New shape to apply to x.

    Returns:
        numpy.ndarray, a new view object or a copy of input array.
    """
    return np.reshape(x, tuple(shp))


def rank(x):
    """
    Gets number of array dimensions.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        int, number of input array dimensions.
    """
    return np.array(np.ndim(x))


def logsoftmax(x):
    """
    Log softmax function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray, the result of applying log softmax on the input array.
    """
    return np.array(np.log(softmax(x)))


def transpose(x, axes=None):
    """
    Transposes an input array according to axes.

    Args:
        x (numpy.ndarray): Input array.
        axes (list): The axes to be transposed. Default: None.

    Returns:
        numpy.ndarray, transposed array.
    """
    return np.transpose(x, axes)


def invert_permutation(x):
    """
    Gets the inverse permutation of an array.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        tuple, the inverse permutation of the input array.
    """
    x = np.array(x)
    y = np.argsort(x)
    return tuple(y)


def select(cond, x, y):
    """
    Gets elements from x or y depending on cond.

    Args:
        cond (bool): Where True, yield x, otherwise yield y.
        x (numpy.ndarray): Values from which to choose.
        y (numpy.ndarray): Values from which to choose.

    Returns:
        numpy.ndarray, elements from x where condition is True, and elements from y elsewhere.
    """
    return np.where(cond, x, y)


def sum_by_axis(x, axis):
    """
    Sum of array elements over a given axis.

    Args:
        x (numpy.ndarray): Input array.
        axis (Union[int, tuple[int]]): Axis or axes along which a sum is performed.

    Returns:
        numpy.ndarray, has the same shape as input array with the specified axis removed.
    """
    return np.sum(x, axis)


def equal(x, y):
    """
    Gets (x == y) element-wise.

    Args:
        x (numpy.ndarray): Input array.
        y (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray, element-wise comparison of x and y.
    """
    return np.equal(x, y)


def not_equal(x, y):
    """
    Gets (x != y) element-wise.

    Args:
        x (numpy.ndarray): Input array.
        y (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray, element-wise comparison of x and y.
    """
    return np.not_equal(x, y)


def greater(x, y):
    """
    Get the truth value of (x > y) element-wise.

    Args:
        x (numpy.ndarray): Input array.
        y (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray, element-wise comparison of x and y.
    """
    return np.greater(x, y)

def less(x, y):
    """
    Get the truth value of (x < y) element-wise.

    Args:
        x (numpy.ndarray): Input array.
        y (numpy.ndarray): Input array.

    Returns:
        Array, element-wise comparison of x and y.
    """
    return np.less(x, y)




def logical_not(x):
    """
    Gets the truth value of NOT x element-wise.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        bool, have the same shape as x of the NOT operation on elements of x.
    """
    return np.logical_not(x)


def sqrt(x):
    """
    Gets the non-negative square-root of an numpy.ndarray, element-wise.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray, has the same shape as x, containing the positive square-root of each
        element in x.
    """
    return np.sqrt(x)


def power(x, y):
    """
    First array elements raised to powers from second numpy.ndarray, element-wise.

    Args:
        x (numpy.ndarray): The bases array.
        y (numpy.ndarray): The exponents array.

    Returns:
        numpy.ndarray, the bases in x raised to the exponents in y.
    """
    return np.power(x, y)


def exp(x):
    """
    Gets the exponential of all elements in the input array.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray, element-wise exponential of x.
    """
    return np.exp(x)


def maximum(x, y):
    """
    Gets the max of x and y element-wise.

    If x > y, return x. Otherwise, return y.

    Args:
        x (numpy.ndarray): First input array.
        y (numpy.ndarray): Second input array ave the same type as x.

    Returns:
        numpy.ndarray, has the same type as x.
    """
    return np.maximum(x, y)


def minimum(x, y):
    """
    Gets the min of x and y element-wise.

    If x < y, return x. Otherwise, return y.

    Args:
        x (numpy.ndarray): First input array.
        y (numpy.ndarray): Second input array have the same type as x.

    Returns:
        numpy.ndarray, has the same type as x.
    """
    return np.minimum(x, y)

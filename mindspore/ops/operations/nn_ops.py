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

"""Operators for nn."""

import math
import operator
from functools import reduce

import numpy as np

from ... import context
from ..._c_expression import signature_rw as sig_rw
from ..._c_expression import signature_kind as sig_kind
from ..._checkparam import ParamValidator as validator
from ..._checkparam import Rel, check_bool, check_int_positive
from ...common import dtype as mstype
from ..primitive import Primitive, PrimitiveWithInfer, prim_attr_register


class Flatten(PrimitiveWithInfer):
    r"""
    Flattens a tensor without changing its batch size on the 0-th axis.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, \ldots)` to be flattened.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, X)`, where :math:`X` is
        the product of the remaining dimension.

    Examples:
        >>> input_tensor = Tensor(np.ones(shape=[1, 2, 3, 4]), mindspore.float32)
        >>> flatten = P.Flatten()
        >>> output = flatten(input_tensor)
        >>> assert output.shape() == (1, 24)
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, input_x):
        validator.check('input_x rank', len(input_x), '', 1, Rel.GE)
        prod = 1 if len(input_x) == 1 else reduce(operator.mul, input_x[1:])
        return input_x[0], prod

    def infer_dtype(self, input_x):
        validator.check_subclass("input_x", input_x, mstype.tensor)
        return input_x


class Softmax(PrimitiveWithInfer):
    r"""
    Softmax operation.

    Applies the Softmax operation to the input tensor on the specified axis.
    Suppose a slice along the given aixs :math:`x` then for each element :math:`x_i`
    the Softmax function is shown as follows:

    .. math::
        \text{output}(x_i) = \frac{exp(x_i)}{\sum_{j = 0}^{N-1}\exp(x_j)},

    where :math:`N` is the length of the tensor.

    Args:
        axis (Union[int, tuple]): The axis to do the Softmax operation. Default: -1.

    Inputs:
        - **logits** (Tensor) - The input of Softmax.

    Outputs:
        Tensor, with the same type and shape as the logits.
    """

    @prim_attr_register
    def __init__(self, axis=-1):
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_type("axis", axis, [int, tuple])
        if isinstance(axis, int):
            self.add_prim_attr('axis', (axis,))
        for item in self.axis:
            validator.check_type("item of axis", item, [int])

    def infer_shape(self, logits):
        validator.check_shape_length("axis shape", len(self.axis), 1, Rel.GE)
        rank = len(logits)
        for axis_v in self.axis:
            validator.check_int_range("axis", axis_v, -rank, rank, Rel.INC_LEFT)
        return logits

    def infer_dtype(self, logits):
        validator.check_subclass("logits", logits, mstype.tensor)
        return logits


class LogSoftmax(PrimitiveWithInfer):
    r"""
    Log Softmax activation function.

    Applies the Log Softmax function to the input tensor on the specified axis.
    Suppose a slice along the given aixs :math:`x` then for each element :math:`x_i`
    the Log Softmax function is shown as follows:

    .. math::
        \text{output}(x_i) = \log \left(\frac{exp(x_i)} {\sum_{j = 0}^{N-1}\exp(x_j)}\right),

    where :math:`N` is the length of the Tensor.

    Args:
        axis (int): The axis to do the Log softmax operation. Default: -1.

    Inputs:
        - **logits** (Tensor) - The input of Log Softmax.

    Outputs:
        Tensor, with the same type and shape as the logits.
    """

    @prim_attr_register
    def __init__(self, axis=-1):
        validator.check_type("axis", axis, [int])

    def infer_shape(self, logits):
        rank = len(logits)
        validator.check_int_range('axis', self.axis, -rank - 1, rank, Rel.INC_BOTH)
        return logits

    def infer_dtype(self, logits):
        validator.check_subclass("logits", logits, mstype.tensor)
        return logits


class ReLU(PrimitiveWithInfer):
    r"""
    Computes ReLU(Rectified Linear Unit) of input tensor element-wise.

    It returns :math:`\max(x,\  0)` element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> relu = P.ReLU()
        >>> result = relu(input_x)
        [[0, 4.0, 0.0], [2.0, 0.0, 9.0]]
    """

    @prim_attr_register
    def __init__(self):
        """init ReLU"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_subclass("input_x", input_x, mstype.tensor)
        validator.check_typename("input_x", input_x, mstype.number_type)
        return input_x


class ReLU6(PrimitiveWithInfer):
    r"""
    Computes ReLU(Rectified Linear Unit) upper bounded by 6 of input tensor element-wise.

    It returns :math:`\min(\max(0,x), 6)` element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> relu6 = P.ReLU6()
        >>> result = relu6(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """init ReLU6"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_subclass("input_x", input_x, mstype.tensor)
        validator.check_typename("input_x", input_x, (mstype.float16, mstype.float32))
        return input_x


class Elu(PrimitiveWithInfer):
    r"""
    Computes exponential linear: `alpha * (exp(x) - 1)` if x < 0, `x` otherwise.
    The data type of input tensor should be float.

    Args:
        alpha (float): The coefficient of negative factor whose type is float. Default: 1.0.

    Inputs:
        - **input_x** (Tensor) - The input tensor whose data type should be float.

    Outputs:
        Tensor, has the same shape and data type as `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> elu = P.Elu()
        >>> result = elu(input_x)
        Tensor([[-0.632  4.0   -0.999]
                [2.0    -0.993  9.0  ]], shape=(2, 3), dtype=mindspore.float32)
    """

    @prim_attr_register
    def __init__(self, alpha=1.0):
        """Init Elu"""
        validator.check_type("alpha", alpha, [float])

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_subclass("input_x", input_x, mstype.tensor)
        validator.check_typename("input_x_dtype", input_x, mstype.float_type)
        return input_x


class HSwish(PrimitiveWithInfer):
    r"""
    Hard swish activation function.

    Applies hswish-type activation element-wise. The input is a Tensor with any valid shape.

    Hard swish is defined as:

    .. math::
        \text{hswish}(x_{i}) = x_{i} * \frac{ReLU6(x_{i} + 3)}{6},

    where :math:`x_{i}` is the :math:`i`-th slice along the given dim of the input Tensor.

    Inputs:
        - **input_data** (Tensor) - The input of Hswish.

    Outputs:
        Tensor, with the same type and shape as the `input_data`.

    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, xshape):
        return xshape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x_dtype", x_dtype, mstype.tensor)
        validator.check_typename("x_dtype", x_dtype, (mstype.float16, mstype.float32))
        return x_dtype



class Sigmoid(PrimitiveWithInfer):
    r"""
    Sigmoid activation function.

    Computes Sigmoid of input element-wise. The Sigmoid function is defined as:

    .. math::
        \text{sigmoid}(x_i) = \frac{1}{1 + exp(-x_i)},

    where :math:`x_i` is the element of the input.

    Inputs:
        - **input_x** (Tensor) - The input of Sigmoid.

    Outputs:
        Tensor, with the same type and shape as the input_x.

    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_subclass("input_x", input_x, mstype.tensor)
        validator.check_typename("input_x", input_x, (mstype.float16, mstype.float32))
        return input_x


class HSigmoid(PrimitiveWithInfer):
    r"""
    Hard sigmoid activation function.

    Applies hard sigmoid activation element-wise. The input is a Tensor with any valid shape.

    Hard sigmoid is defined as:

    .. math::
        \text{hsigmoid}(x_{i}) = max(0, min(1, \frac{2 * x_{i} + 5}{10})),

    where :math:`x_{i}` is the :math:`i`-th slice along the given dim of the input Tensor.

    Inputs:
        - **input_data** (Tensor) - The input of HSigmoid.

    Outputs:
        Tensor, with the same type and shape as the `input_data`.

    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("x_dtype", x_dtype, mstype.tensor)
        validator.check_typename("x_dtype", x_dtype, (mstype.float16, mstype.float32))
        return x_dtype


class Tanh(PrimitiveWithInfer):
    r"""
    Tanh activation function.

    Computes hyperbolic tangent of input element-wise. The Tanh function is defined as:

    .. math::
        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    where :math:`x_i` is an element of the input Tensor.

    Inputs:
        - **input_x** (Tensor) - The input of Tanh.

    Outputs:
        Tensor, with the same type and shape as the input_x.
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_subclass("input_x", input_x, mstype.tensor)
        return input_x


class FusedBatchNorm(Primitive):
    r"""
    FusedBatchNorm is a BatchNorm that moving mean and moving variance will be computed instead of being loaded.

    Batch Normalization is widely used in convolutional networks. This operation applies
    Batch Normalization over input to avoid internal covariate shift as described in the
    paper `Batch Normalization: Accelerating Deep Network Training by Reducing Internal
    Covariate Shift <https://arxiv.org/abs/1502.03167>`_. It rescales and recenters the
    feature using a mini-batch of data and the learned parameters which can be described
    in the following formula.

    .. math::
        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    where :math:`\gamma` is scale, :math:`\beta` is bias, :math:`\epsilon` is epsilon.

    Args:
        mode (int): Mode of batch normalization, value is 0 or 1. Default: 0.
        epsilon (float): A small value added for numerical stability. Default: 1e-5.
        momentum (float): The hyper parameter to compute moving average for running_mean and running_var
            (e.g. :math:`new\_running\_mean = momentum * running\_mean + (1 - momentum) * current\_mean`).
            Momentum value should be [0, 1]. Default: 0.9.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, C)`.
        - **scale** (Tensor) - Tensor of shape :math:`(C,)`.
        - **bias** (Tensor) - Tensor of shape :math:`(C,)`.
        - **mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **variance** (Tensor) - Tensor of shape :math:`(C,)`.

    Outputs:
        Tuple of 5 Tensor, the normalized input and the updated parameters.

        - **output_x** (Tensor) - The same type and shape as the `input_x`.
        - **updated_scale** (Tensor) - Tensor of shape :math:`(C,)`.
        - **updated_bias** (Tensor) - Tensor of shape :math:`(C,)`.
        - **updated_moving_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **updated_moving_variance** (Tensor) - Tensor of shape :math:`(C,)`.
    """

    @prim_attr_register
    def __init__(self, mode=0, epsilon=1e-5, momentum=0.1):
        self.init_prim_io_names(inputs=['x', 'scale', 'b', 'mean', 'variance'],
                                outputs=['y', 'running_mean', 'running_variance', 'save_mean', 'save_inv_variance'])
        self.mode = validator.check_integer('mode', mode, [0, 1], Rel.IN)
        self.epsilon = validator.check_number_range('epsilon', epsilon, 0, 1, Rel.INC_RIGHT)
        self.momentum = validator.check_number_range('momentum', momentum, 0, 1, Rel.INC_BOTH)


class BatchNorm(PrimitiveWithInfer):
    r"""
    Batch Normalization for input data and updated parameters.

    Batch Normalization is widely used in convolutional neural networks. This operation
    applies Batch Normalization over input to avoid internal covariate shift as described
    in the paper `Batch Normalization: Accelerating Deep Network Training by Reducing Internal
    Covariate Shift <https://arxiv.org/abs/1502.03167>`_. It rescales and recenters the
    features using a mini-batch of data and the learned parameters which can be described
    in the following formula,

    .. math::
        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    where :math:`\gamma` is scale, :math:`\beta` is bias, :math:`\epsilon` is epsilon.

    Args:
        is_training (bool): If `is_training` is True, `mean` and `variance` are computed during training.
            If `is_training` is False, they're loaded from checkpoint during inference. Default: False.
        epsilon (float): A small value added for numerical stability. Default: 1e-5.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, C)`.
        - **scale** (Tensor) - Tensor of shape :math:`(C,)`.
        - **bias** (Tensor) - Tensor of shape :math:`(C,)`.
        - **mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **variance** (Tensor) - Tensor of shape :math:`(C,)`.

    Outputs:
        Tuple of 5 Tensor, the normalized inputs and the updated parameters.

        - **output_x** (Tensor) - The same type and shape as the input_x. The shape is :math:`(N, C)`.
        - **updated_scale** (Tensor) - Tensor of shape :math:`(C,)`.
        - **updated_bias** (Tensor) - Tensor of shape :math:`(C,)`.
        - **reserve_space_1** (Tensor) - Tensor of shape :math:`(C,)`.
        - **reserve_space_2** (Tensor) - Tensor of shape :math:`(C,)`.
        - **reserve_space_3** (Tensor) - Tensor of shape :math:`(C,)`.
    """

    @prim_attr_register
    def __init__(self, is_training=False, epsilon=1e-5):
        self.is_training = validator.check_type('is_training', is_training, (bool,))
        self.epsilon = validator.check_number_range('epsilon', epsilon, 0, 1, Rel.INC_RIGHT)
        self.add_prim_attr('data_format', "NCHW")
        self.init_prim_io_names(inputs=['x', 'scale', 'offset', 'mean', 'variance'],
                                outputs=['y', 'batch_mean', 'batch_variance', 'reserve_space_1', 'reserve_space_2',
                                         'reserve_space_3'])

    def infer_shape(self, input_x, scale, bias, mean, variance):
        validator.check("BatchNorm scale shape length", len(scale), "1", 1, Rel.EQ)
        validator.check("BatchNorm scale shape", scale, "BatchNorm bias shape", bias)
        validator.check("BatchNorm scale shape", scale[0], "BatchNorm input_x shape[1]", input_x[1])
        if not self.is_training:
            validator.check("BatchNorm mean shape length", len(mean), "1", 1, Rel.EQ)
            validator.check("BatchNorm mean shape", mean, "BatchNorm variance shape", variance)
            validator.check("BatchNorm mean shape", mean, "BatchNorm scale shape", scale)
        return (input_x, scale, scale, scale, scale, scale)

    def infer_dtype(self, input_x, scale, bias, mean, variance):
        args = {"BatchNorm scale type": scale, "BatchNorm bias type": bias}
        args_moving = {"BatchNorm mean type": mean, "BatchNorm variance type": variance}
        validator.check_typename("input_x", input_x, [mstype.float32, mstype.float16])
        validator.check_type_same(args, [mstype.float32, mstype.float16])
        if self.is_training:
            validator.check_type_same(args_moving, [mstype.float32, mstype.float16, None])
        else:
            validator.check_type_same(args_moving, [mstype.float32, mstype.float16])
        return (input_x, scale, bias, input_x, input_x, input_x)


class Conv2D(PrimitiveWithInfer):
    r"""
    2D convolution layer.

    Applies a 2D convolution over an input tensor which is typically of shape :math:`(N, C_{in}, H_{in}, W_{in})`,
    where :math:`N` is batch size and :math:`C_{in}` is channel number. For each batch of shape
    :math:`(C_{in}, H_{in}, W_{in})`, the formula is defined as:

    .. math::

        out_j = \sum_{i=0}^{C_{in} - 1} ccor(W_{ij}, X_i) + b_j,

    where :math:`ccor` is cross correlation operator, :math:`C_{in}` is the input channel number, :math:`j` ranges
    from :math:`0` to :math:`C_{out} - 1`, :math:`W_{ij}` corresponds to :math:`i`-th channel of the :math:`j`-th
    filter and :math:`out_{j}` corresponds to the :math:`j`-th channel of the output. :math:`W_{ij}` is a slice
    of kernel and it has shape :math:`(\text{ks_h}, \text{ks_w})`, where :math:`\text{ks_h}` and
    :math:`\text{ks_w}` are height and width of the convolution kernel. The full kernel has shape
    :math:`(C_{out}, C_{in} // \text{group}, \text{ks_h}, \text{ks_w})`, where group is the group number
    to split the input in the channel dimension.

    If the 'pad_mode' is set to be "valid", the output height and width will be
    :math:`\left \lfloor{1 + \frac{H_{in} + 2 \times \text{padding} - \text{ks_h} -
    (\text{ks_h} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` and
    :math:`\left \lfloor{1 + \frac{W_{in} + 2 \times \text{padding} - \text{ks_w} -
    (\text{ks_w} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` respectively.


    The first introduction can be found in paper `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_. More detailed introduction can be found here:
    http://cs231n.github.io/convolutional-networks/.

    Args:
        out_channel (int): The dimension of the output.
        kernel_size (Union[int, tuple[int]]): The kernel size of the 2D convolution.
        mode (int): 0 Math convolutiuon, 1 cross-correlation convolution ,
                       2 deconvolution, 3 depthwise convolution. Default: 1.
        pad_mode (str): "valid", "same", "pad" the mode to fill padding. Default: "valid".
        pad (int): The pad value to fill. Default: 0.
        stride (Union(int, tuple[int])): The stride to apply conv filter. Default: 1.
        dilation (Union(int, tuple[int])): Specify the space to use between kernel elements. Default: 1.
        group (int): Split input into groups. Default: 1.

    Returns:
        Tensor, the value that applied 2D convolution.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.
        - **weight** (Tensor) - Set size of kernel is :math:`(K_1, K_2)`, then the shape is
          :math:`(C_{out}, C_{in}, K_1, K_2)`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.
    """

    @prim_attr_register
    def __init__(self,
                 out_channel,
                 kernel_size,
                 mode=1,
                 pad_mode="valid",
                 pad=0,
                 stride=1,
                 dilation=1,
                 group=1):
        """init Conv2D"""
        self.init_prim_io_names(inputs=['x', 'w'], outputs=['output'])
        self.kernel_size = validator.check_type('kernel_size', kernel_size, (int, tuple))
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        if len(self.kernel_size) != 2 or (not isinstance(self.kernel_size[0], int)) or \
                                         (not isinstance(self.kernel_size[1], int)) or \
                                         self.kernel_size[0] < 1 or self.kernel_size[1] < 1:
            raise ValueError(f"The \'kernel_size\' of \'Conv2D\' should be an positive int number or "
                             f"a tuple of two positive int numbers, but got {kernel_size}")
        self.stride = validator.check_type('stride', stride, (int, tuple))
        if isinstance(stride, int):
            self.stride = (stride, stride)
        if len(self.stride) != 2 or (not isinstance(self.stride[0], int)) or \
                                    (not isinstance(self.stride[1], int)) or \
                                     self.stride[0] < 1 or self.stride[1] < 1:
            raise ValueError(f"The \'stride\' of \'Conv2D\' should be an positive int number or "
                             f"a tuple of two positive int numbers, but got {stride}")
        self.add_prim_attr('stride', (1, 1, self.stride[0], self.stride[1]))
        self.dilation = validator.check_type('dilation', dilation, (tuple, int))
        if isinstance(dilation, int):
            self.dilation = (1, 1, dilation, dilation)
        elif len(dilation) == 2:
            self.dilation = (1, 1, dilation[0], dilation[1])
        if len(self.dilation) != 4 or (not isinstance(self.dilation[0], int) or self.dilation[0] < 1) or \
                                      (not isinstance(self.dilation[1], int) or self.dilation[1] < 1) or \
                                      (not isinstance(self.dilation[2], int) or self.dilation[2] < 1) or \
                                      (not isinstance(self.dilation[3], int) or self.dilation[3] < 1):
            raise ValueError(f"The \'dilation\' of \'Conv2D\' should be an positive int number or "
                             f"a tuple of two or four positive int numbers, but got {dilation}")
        self.add_prim_attr('dilation', self.dilation)
        validator.equal('type of pad', type(pad), 'not bool', not isinstance(pad, bool))
        validator.equal('type of pad', type(pad), 'int', isinstance(pad, int))
        self.pad_mode = validator.check_string('pad_mode', pad_mode, ['valid', 'same', 'pad'])
        self.pad = validator.check_pad_value_by_mode(self.__class__.__name__, pad_mode, pad)
        if self.pad_mode == 'pad':
            validator.check_integer('pad', self.pad, 0, Rel.GE)

        self.mode = validator.check_integer('mode', mode, 1, Rel.EQ)
        self.add_prim_attr('data_format', "NCHW")
        self.out_channel = validator.check_integer('out_channel', out_channel, 0, Rel.GT)
        self.group = validator.check_integer('group', group, 0, Rel.GT)

    def infer_shape(self, x_shape, w_shape):
        validator.check_integer("weight_shape", len(w_shape), 4, Rel.EQ)
        validator.check_integer("x_shape", len(x_shape), 4, Rel.EQ)
        validator.check_param_equal("x_shape[1]", x_shape[1] // self.group, "w_shape[1]", w_shape[1])
        validator.check_param_equal('out_channel', self.out_channel, 'w_shape[0]', w_shape[0])
        validator.check_param_equal('kernel_size', self.kernel_size, 'w_shape[2:4]', tuple(w_shape[2:4]))

        kernel_size_h = w_shape[2]
        kernel_size_w = w_shape[3]
        stride_h = self.stride[2]
        stride_w = self.stride[3]
        dilation_h = self.dilation[2]
        dilation_w = self.dilation[3]

        if self.pad_mode == "valid":
            h_out = math.ceil((x_shape[2] - dilation_h * (kernel_size_h - 1)) / stride_h)
            w_out = math.ceil((x_shape[3] - dilation_w * (kernel_size_w - 1)) / stride_w)
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        elif self.pad_mode == "same":
            h_out = math.ceil(x_shape[2] / stride_h)
            w_out = math.ceil(x_shape[3] / stride_w)

            pad_needed_h = max(0, (h_out - 1) * stride_h + dilation_h * (kernel_size_h - 1) + 1 - x_shape[2])
            pad_top = math.floor(pad_needed_h / 2)
            pad_bottom = pad_needed_h - pad_top

            pad_needed_w = max(0, (w_out - 1) * stride_w + dilation_w * (kernel_size_w - 1) + 1 - x_shape[3])
            pad_left = math.floor(pad_needed_w / 2)
            pad_right = pad_needed_w - pad_left
        elif self.pad_mode == 'pad':
            pad_top, pad_bottom, pad_left, pad_right = self.pad, self.pad, self.pad, self.pad

            h_out = 1 + (x_shape[2] + 2 * self.pad - kernel_size_h - (kernel_size_h - 1) * (dilation_h - 1)) \
                    / stride_h
            w_out = 1 + (x_shape[3] + 2 * self.pad - kernel_size_w - (kernel_size_w - 1) * (dilation_w - 1)) \
                    / stride_w
            h_out = math.floor(h_out)
            w_out = math.floor(w_out)

        self.pad_list = [pad_top, pad_bottom, pad_left, pad_right]
        self.add_prim_attr('pad_list', (pad_top, pad_bottom, pad_left, pad_right))

        out_channel = self.out_channel
        out_shape = [x_shape[0], out_channel, h_out, w_out]
        return out_shape

    def infer_dtype(self, x_dtype, w_dtype):
        args = {'x_dtype': x_dtype, 'w_dtype': w_dtype}
        validator.check_subclass('input', x_dtype, mstype.tensor)
        validator.check_subclass('weight', w_dtype, mstype.tensor)
        validator.check_type_same(args, [mstype.int8, mstype.int32, mstype.float16, mstype.float32])
        return x_dtype


class DepthwiseConv2dNative(PrimitiveWithInfer):
    r"""
    Returns the depth-wise convolution value for the input.

    Applies depthwise conv2d for the input, which will generate more channels with channel_multiplier.
    Given an input tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})` where :math:`N` is the batch size and a
    filter tensor with kernel size :math:`(ks_{h}, ks_{w})`, containing :math:`C_{in} * \text{channel_multiplier}`
    convolutional filters of depth 1; it applies different filters to each input channel (channel_multiplier channels
    for each with default value 1), then concatenates the results together. The output has
    :math:`\text{in_channels} * \text{channel_multiplier}` channels.

    Args:
        channel_multiplier (int): The multipiler for the original output conv.
        kernel_size (Union[int, tuple[int]]): The size of the conv kernel.
        mode (int): 0 Math convolution, 1 cross-correlation convolution ,
                       2 deconvolution, 3 depthwise convolution. Default: 3.
        pad_mode (str): "valid", "same", "pad" the mode to fill padding. Default: "valid".
        pad (int): The pad value to fill. Default: 0.
        stride (Union[int, tuple[int]]): The stride to apply conv filter. Default: 1.
        dilation (Union[int, tuple[int]]): Specifies the dilation rate to use for dilated convolution. Default: 1.
        group (int): Splits input into groups. Default: 1.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.
        - **weight** (Tensor) - Set size of kernel is :math:`(K_1, K_2)`, then the shape is
          :math:`(channel_{multiplier}, C_{in}, K_1, K_2)`.

    Outputs:
        Tensor of shape :math:`(N, C_{in} * \text{channel_multiplier}, H_{out}, W_{out})`.
    """

    @prim_attr_register
    def __init__(self,
                 channel_multiplier,
                 kernel_size,
                 mode=3,
                 pad_mode="valid",
                 pad=0,
                 stride=1,
                 dilation=1,
                 group=1):
        """init DepthwiseConv2dNative"""
        validator.check_pad_value_by_mode(self.__class__.__name__, pad_mode, pad)
        self.kernel_size = validator.check_type('kernel_size', kernel_size, (int, tuple))
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        if len(self.kernel_size) != 2 or (not isinstance(self.kernel_size[0], int)) or \
                                         (not isinstance(self.kernel_size[1], int)) or \
                                         self.kernel_size[0] < 1 or self.kernel_size[1] < 1:
            raise ValueError(f"The \'kernel_size\' of \'DepthwiseConv2dNative\' should be an positive int number or "
                             f"a tuple of two positive int numbers, but got {kernel_size}")
        self.stride = validator.check_type('stride', stride, (int, tuple))
        if isinstance(stride, int):
            self.stride = (stride, stride)
        if len(self.stride) != 2 or (not isinstance(self.stride[0], int)) or \
                                    (not isinstance(self.stride[1], int)) or \
                                    self.stride[0] < 1 or self.stride[1] < 1:
            raise ValueError(f"The \'stride\' of \'DepthwiseConv2dNative\' should be an positive int number or "
                             f"a tuple of two positive int numbers, but got {stride}")
        self.add_prim_attr('stride', (1, 1, self.stride[0], self.stride[1]))
        self.dilation = validator.check_type('dilation', dilation, (tuple, int))
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        if len(self.dilation) != 2 or (not isinstance(self.dilation[0], int)) or \
                                      (not isinstance(self.dilation[1], int)) or \
                                      self.dilation[0] < 1 or self.dilation[1] < 1:
            raise ValueError(f"The \'dilation\' of \'DepthwiseConv2dNative\' should be an positive int number or "
                             f"a tuple of two or four positive int numbers, but got {dilation}")
        self.add_prim_attr('dilation', (1, 1, self.dilation[0], self.dilation[1]))
        validator.equal('type of pad', type(pad), 'not bool', not isinstance(pad, bool))
        if pad_mode not in ("same", "valid", "pad"):
            raise ValueError(f"Attr pad_mode of DepthwiseConv2dNative Op not passed"
                             f"{pad_mode} not in valid, same, pad.")
        self.pad_mode = pad_mode
        self.mode = validator.check_integer("mode", mode, 3, Rel.EQ)
        self.add_prim_attr('data_format', "NCHW")
        self.channel_multiplier = validator.check_integer("channel_multiplier", channel_multiplier, 0, Rel.GT)
        self.group = validator.check_integer("group", group, 0, Rel.GT)
        self.pad = pad

    def infer_shape(self, x_shape, w_shape):
        validator.check_integer("weight_shape", len(w_shape), 4, Rel.EQ)
        validator.check_integer("x_shape", len(x_shape), 4, Rel.EQ)
        validator.check_param_equal("x_shape[1]", x_shape[1], "w_shape[1]", w_shape[1])
        validator.check_param_equal('kernel_size', self.kernel_size, 'w_shape[2:4]', tuple(w_shape[2:4]))

        kernel_size_h = w_shape[2]
        kernel_size_w = w_shape[3]
        stride_h = self.stride[2]
        stride_w = self.stride[3]
        dilation_h = self.dilation[2]
        dilation_w = self.dilation[3]

        if self.pad_mode == "valid":
            h_out = math.ceil((x_shape[2] - dilation_h * (kernel_size_h - 1)) / stride_h)
            w_out = math.ceil((x_shape[3] - dilation_w * (kernel_size_w - 1)) / stride_w)
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        elif self.pad_mode == "same":
            h_out = math.ceil(x_shape[2] / stride_h)
            w_out = math.ceil(x_shape[3] / stride_w)

            pad_needed_h = max(0, (h_out - 1) * stride_h+ dilation_h * (kernel_size_h - 1) + 1 - x_shape[2])
            pad_top = math.floor(pad_needed_h / 2)
            pad_bottom = pad_needed_h - pad_top

            pad_needed_w = max(0, (w_out - 1) * stride_w + dilation_w * (kernel_size_w - 1) + 1 - x_shape[3])
            pad_left = math.floor(pad_needed_w / 2)
            pad_right = pad_needed_w - pad_left
        elif self.pad_mode == 'pad':
            pad_top, pad_bottom, pad_left, pad_right = self.pad, self.pad, self.pad, self.pad

            h_out = 1 + (x_shape[2] + 2 * self.pad - kernel_size_h - (kernel_size_h - 1) * (dilation_h - 1)) \
                    / stride_h
            w_out = 1 + (x_shape[3] + 2 * self.pad - kernel_size_w - (kernel_size_w - 1) * (dilation_w - 1)) \
                    / stride_w
            h_out = math.floor(h_out)
            w_out = math.floor(w_out)
        else:
            raise ValueError(f"Attr pad_mode of DepthwiseConv2dNative Op not passed"
                             "{pad_mode} not in valid, same, pad.")

        self.pad_list = (pad_top, pad_bottom, pad_left, pad_right)
        self.add_prim_attr('pads', self.pad_list)

        out_channel = self.channel_multiplier * x_shape[1]
        out_shape = [x_shape[0], out_channel, h_out, w_out]
        return out_shape

    def infer_dtype(self, x_dtype, w_dtype):
        args = {'x_dtype': x_dtype, 'w_dtype': w_dtype}
        validator.check_type_same(args, mstype.number_type)
        return x_dtype


class _Pool(PrimitiveWithInfer):
    r"""
    Performs max/avg pooling operation.

    Args:
        ksize (Union[int, tuple[int]]): The size of the kernel, that should be a tuple
           of two `int` for height and width. Default: 1.
        strides (Union[int, tuple[int]]): The stride of the window, that should be
            a tuple of two `int` for height and width. Default: 1.
        padding (str): The optional values for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".
    """

    @prim_attr_register
    def __init__(self, ksize=1, strides=1, padding="valid"):
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_type('ksize', ksize, [int, tuple])
        validator.check_type('strides', strides, [int, tuple])
        self.padding = validator.check_string('padding', padding.upper(), ['VALID', 'SAME'])
        self.add_prim_attr("padding", self.padding)
        self.is_maxpoolwithargmax = (self.name == "MaxPoolWithArgmax")
        if not self.is_maxpoolwithargmax:
            self.add_prim_attr('data_format', "NCHW")

        if isinstance(ksize, int):
            validator.check_integer("ksize", ksize, 1, Rel.GE)
            self.ksize = (1, 1, ksize, ksize)
        else:
            if (len(ksize) != 2 or
                    (not isinstance(ksize[0], int)) or
                    (not isinstance(ksize[1], int)) or
                    ksize[0] <= 0 or
                    ksize[1] <= 0):
                raise ValueError(f"The 'ksize' passed to operator {self.name} should be an positive int number or "
                                 f"a tuple of two positive int numbers, but got {ksize}")
            self.ksize = (1, 1, ksize[0], ksize[1])
        if self.is_maxpoolwithargmax:
            self.ksize = (1, self.ksize[-2], self.ksize[-1], 1)
        self.add_prim_attr("ksize", self.ksize)

        if isinstance(strides, int):
            validator.check_integer("strides", strides, 1, Rel.GE)
            self.strides = (1, 1, strides, strides)
        else:
            if (len(strides) != 2 or
                    (not isinstance(strides[0], int)) or
                    (not isinstance(strides[1], int)) or
                    strides[0] <= 0 or
                    strides[1] <= 0):
                raise ValueError(f"The 'strides' passed to operator {self.name} should be an positive int number or "
                                 f"a tuple of two positive int numbers, but got {strides}")
            self.strides = (1, 1, strides[0], strides[1])
        if self.is_maxpoolwithargmax:
            self.strides = (1, self.strides[-2], self.strides[-1], 1)
        self.add_prim_attr("strides", self.strides)

    def infer_shape(self, x_shape):
        validator.check_integer("x_shape", len(x_shape), 4, Rel.EQ)
        batch, channel, input_h, input_w = x_shape
        if self.is_maxpoolwithargmax:
            _, kernel_h, kernel_w, _ = self.ksize
            _, stride_h, stride_w, _ = self.strides
        else:
            _, _, kernel_h, kernel_w = self.ksize
            _, _, stride_h, stride_w = self.strides

        if self.padding == "VALID":
            out_h = math.ceil((input_h - (kernel_h - 1)) / stride_h)
            out_w = math.ceil((input_w - (kernel_w - 1)) / stride_w)
        elif self.padding == "SAME":
            out_h = math.ceil(input_h / stride_h)
            out_w = math.ceil(input_w / stride_w)
        else:
            raise ValueError(f"The padding of operator {self.name} should be a str and must be 'SAME' or 'VALID', "
                             f"but got {self.padding}.")
        out_shape = [batch, channel, out_h, out_w]

        for shape_value in out_shape:
            if shape_value <= 0:
                raise ValueError("The kernel size is not valid please check it if is larger than data's shape size.")
        return out_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input", x_dtype, mstype.tensor)
        return x_dtype


class MaxPool(_Pool):
    r"""
    Max pooling operation.

    Applies a 2D max pooling over an input Tensor which can be regarded as a composition of 2D planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, MaxPool outputs
    regional maximum in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    Args:
        ksize (Union[int, tuple[int]]): The size of kernel used to take the maximum value,
            is an int number that represents height and width are both ksize, or a tuple
            of two int numbers that represent height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        padding (str): The optional values for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. Output height and width will be the same as
              the input. Total number of padding will be calculated for horizontal and vertical
              direction and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possibly largest height and width of output
              will be return without padding. Extra pixels will be discarded.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, with shape :math:`(N, C_{out}, H_{out}, W_{out})`.
    """

    @prim_attr_register
    def __init__(self, ksize=1, strides=1, padding="valid"):
        super(MaxPool, self).__init__(ksize, strides, padding)


class MaxPoolWithArgmax(_Pool):
    r"""
    Performs max pooling on the input Tensor and return both max values and indices.

    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, MaxPool outputs
    regional maximum in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    Args:
        ksize (Union[int, tuple[int]]): The size of kernel used to take the maximum value and arg value,
            is an int number that represents height and width are both ksize, or a tuple of
            two int numbers that represent height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        padding (str): The optional values for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. Output height and width will be the same as
              the input. Total number of padding will be calculated for horizontal and vertical
              direction and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possibly largest height and width of output
              will be return without padding. Extra pixels will be discarded.


    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tuple of 2 Tensor, the maxpool result and where max values from.

        - **output** (Tensor) -  Maxpooling result, with shape :math:`(N, C_{out}, H_{out}, W_{out})`.
        - **mask** (Tensor) -  Max values' index represented by the mask.
    """
    def __init__(self, ksize=1, strides=1, padding="valid"):
        super(MaxPoolWithArgmax, self).__init__(ksize, strides, padding)
        self.is_tbe = context.get_context("device_target") == "Ascend"

    def infer_shape(self, x_shape):
        out_shape = _Pool.infer_shape(self, x_shape)
        _, _, out_h, out_w = out_shape
        _, kernel_h, kernel_w, _ = self.ksize

        argmax_shape = []
        if self.is_tbe:
            for i in range(4):
                if i == 2:
                    dim = kernel_h * kernel_w
                    argmax_shape.append(dim)
                elif i == 3:
                    dim = math.ceil(out_h * out_w / 16) + 1
                    argmax_shape.append(dim)
                else:
                    argmax_shape.append(x_shape[i])
        else:
            argmax_shape = out_shape

        return out_shape, argmax_shape

    def infer_dtype(self, x_dtype):
        out_dtype = x_dtype
        validator.check_typename("x_type", x_dtype, (mstype.float16, mstype.float32))
        argmax_dtype = mstype.uint16
        return out_dtype, argmax_dtype


class AvgPool(_Pool):
    r"""
    Average pooling operation.

    Applies a 2D average pooling over an input Tensor which can be regarded as a composition of 2D input planes.
    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, AvgPool2d outputs
    regional average in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \frac{1}{h_{ker} * w_{ker}} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    Args:
        ksize (Union[int, tuple[int]]): The size of kernel used to take the average value,
            is an int number that represents height and width are both ksize, or a tuple
            of two int numbers that represent height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        padding (str): The optional values for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. Output height and width will be the same as
              the input. Total number of padding will be calculated for horizontal and vertical
              direction and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possibly largest height and width of output
              will be return without padding. Extra pixels will be discarded.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, with shape :math:`(N, C_{out}, H_{out}, W_{out})`.
    """

    @prim_attr_register
    def __init__(self, ksize=1, strides=1, padding="valid"):
        if context.get_context("device_target") == "GPU":
            self.target = "GPU"
        else:
            self.target = "OTHER"
        super(AvgPool, self).__init__(ksize, strides, padding)


class Conv2DBackpropInput(PrimitiveWithInfer):
    """
    Computes the gradients of convolution with respect to the input.

    Args:
        out_channel (int): The dimensionality of the output space.
        kernel_size (Union[int, tuple[int]]): The size of the convolution window.
        pad_mode (str): "valid", "same", "pad" the mode to fill padding. Default: "valid".
        pad (int): The pad value to fill. Default: 0.
        mode (int): 0 Math convolutiuon, 1 cross-correlation convolution ,
                       2 deconvolution, 3 depthwise convolution. Default: 1.
        stride (Union[int. tuple[int]]): The stride to apply conv filter. Default: 1.
        dilation (Union[int. tuple[int]]): Specifies the dilation rate to use for dilated convolution. Default: 1.
        group (int): Splits input into groups. Default: 1.

    Returns:
        Tensor, the gradients of convolution.
    """

    @prim_attr_register
    def __init__(self,
                 out_channel,
                 kernel_size,
                 pad_mode="valid",
                 pad=0,
                 pad_list=None,
                 mode=1,
                 stride=1,
                 dilation=1,
                 group=1):
        """init Conv2DBackpropInput"""
        self.init_prim_io_names(inputs=['out_backprop', 'filter', 'input_sizes'], outputs=['output'])
        self.out_channel = validator.check_integer('out_channel', out_channel, 0, Rel.GT)
        self.kernel_size = validator.check_type('kernel_size', kernel_size, (int, tuple))
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        if len(self.kernel_size) != 2 or (not isinstance(self.kernel_size[0], int)) or \
                                         (not isinstance(self.kernel_size[1], int)) or \
                                         self.kernel_size[0] < 1 or self.kernel_size[1] < 1:
            raise ValueError(f"The \'kernel_size\' of \'Conv2DBackpropInput\' should be an positive int number or "
                             f"a tuple of two positive int numbers, but got {kernel_size}")
        self.stride = validator.check_type('stride', stride, (int, tuple))
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple) and len(stride) == 4:
            self.stride = (stride[2], stride[3])
        if len(self.stride) != 2 or (not isinstance(self.stride[0], int)) or (not isinstance(self.stride[1], int)) or \
                self.stride[0] < 1 or self.stride[1] < 1:
            raise ValueError(f"The \'stride\' of \'Conv2DBackpropInput\' should be an positive int number or "
                             f"a tuple of two or four positive int numbers, but got {stride}")
        self.add_prim_attr('stride', self.stride)
        self.dilation = validator.check_type('dilation', dilation, (tuple, int))
        if isinstance(dilation, int):
            self.dilation = (1, 1, dilation, dilation)
        elif len(dilation) == 2:
            self.dilation = (1, 1, dilation[0], dilation[1])
        if len(self.dilation) != 4 or (not isinstance(self.dilation[0], int) or self.dilation[0] < 1) or \
                                      (not isinstance(self.dilation[1], int) or self.dilation[1] < 1) or \
                                      (not isinstance(self.dilation[2], int) or self.dilation[2] < 1) or \
                                      (not isinstance(self.dilation[3], int) or self.dilation[3] < 1):
            raise ValueError(f"The \'dilation\' of \'Conv2DBackpropInput\' should be an positive int number or "
                             f"a tuple of two or four positive int numbers, but got {dilation}")
        self.add_prim_attr('dilation', self.dilation)
        validator.equal('type of pad', type(pad), 'not bool', not isinstance(pad, bool))
        validator.equal('type of pad', type(pad), 'int', isinstance(pad, int))
        self.pad_mode = validator.check_string('pad_mode', pad_mode, ['valid', 'same', 'pad'])
        self.pad = validator.check_pad_value_by_mode(self.__class__.__name__, pad_mode, pad)
        self.mode = validator.check_integer('mode', mode, 1, Rel.EQ)
        self.group = validator.check_integer('group', group, 0, Rel.GT)
        pad_mode = pad_mode.upper()
        self.add_prim_attr('pad_mode', pad_mode)
        self.add_prim_attr('data_format', "NCHW")
        if pad_list:
            self.pad_lsit = (validator.check_integer('pad_list', x, 0, Rel.GE) for x in pad_list)

    def __infer__(self, doutput, w, x_size):
        x_size_v = x_size['value']
        validator.check_type('x_size', x_size_v, [tuple])
        for i, dim_len in enumerate(x_size_v):
            validator.check_type("x_size[%d]" % i, dim_len, [int])
        validator.check_typename('w_dtype', w['dtype'], [mstype.int8, mstype.int32, mstype.float16, mstype.float32])
        validator.check_two_types_same('doutput_dtype', doutput['dtype'], 'w_dtype', w['dtype'])

        # infer shape
        dout_shape = doutput['shape']
        kernel_h = self.kernel_size[0]
        kernel_w = self.kernel_size[1]
        stride_h = self.stride[0]
        stride_w = self.stride[1]
        # default pad mode is valid
        pad_list = (0, 0, 0, 0)
        if self.pad_list:
            pad_list = tuple(self.pad_list)
        elif self.pad_mode == "SAME":
            pad_needed_h = max(0, (dout_shape[2] - 1) * stride_h + kernel_h - x_size_v[2])
            pad_top = math.floor(pad_needed_h / 2)
            pad_bottom = pad_needed_h - pad_top

            pad_needed_w = max(0, (dout_shape[3] - 1) * stride_w + kernel_w - x_size_v[3])
            pad_left = math.floor(pad_needed_w / 2)
            pad_right = pad_needed_w - pad_left
            pad_list = (pad_top, pad_bottom, pad_left, pad_right)
        elif self.pad_mode == 'PAD':
            pad_list = (self.pad,) * 4
        self.add_prim_attr('pad_list', pad_list)
        out = {
            'value': None,
            'shape': x_size_v,
            'dtype': doutput['dtype'],
        }
        return out


class BiasAdd(PrimitiveWithInfer):
    r"""
    Returns sum of input and bias tensor.

    Adds the 1-D bias tensor to the input tensor, and boardcasts the shape on all axis
    except for the channel axis.

    Inputs:
        - **input_x** (Tensor) - Input value, with shape :math:`(N, C)` or :math:`(N, C, H, W)`.
        - **bias** (Tensor) - Bias value, with shape :math:`(C)`.

    Outputs:
        Tensor, with the same shape and type as `input_x`.
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'b'], outputs=['output'])
        self.add_prim_attr('data_format', 'NCHW')

    def infer_shape(self, x_shape, b_shape):
        if len(b_shape) != 1 or len(x_shape) < 2 or b_shape[0] != x_shape[1]:
            raise ValueError("Input_x and bias shapes do not match",
                             "(require: rank of input_x must be at least 2, rank of bias must be 1, "
                             "input_x.dim[1] must equal bias.dim[0]),"
                             " but got input_x shape {}, bias shape {}.".format(x_shape, b_shape))
        return x_shape

    def infer_dtype(self, x_type, b_type):
        args = {"input_x type": x_type, "bias type": b_type}
        validator.check_type_same(args, (mstype.float16, mstype.float32, mstype.int8, mstype.int32))
        return x_type


class TopK(PrimitiveWithInfer):
    """
    Finds values and indices of the `k` largest entries along the last dimension.

    Args:
        sorted (bool): If true, the resulting elements will
            be sorted by the values in descending order. Default: False.

    Inputs:
        - **input_x** (Tensor) - Input to be computed.
        - **k** (int) - Number of top elements to be computed along the last dimension, constant input is needed.

    Outputs:
        Tuple of 2 Tensor, the values and the indices.

        - **values** (Tensor) - The `k` largest elements along each last dimensional slice.
        - **indices** (Tensor) - The indices of values within the last dimension of input.

    Examples:
        >>> topk = P.TopK(sorted=True)
        >>> input_x = Tensor([1, 2, 3, 4, 5], mindspore.float16)
        >>> k = 3
        >>> values, indices = topk(input_x, k)
        >>> assert values == Tensor(np.array([5, 4, 3]))
        >>> assert indices == Tensor(np.array([4, 3, 2]))
    """

    @prim_attr_register
    def __init__(self, sorted=False):
        validator.check_type("sorted", sorted, [bool])
        self.init_prim_io_names(inputs=['input', 'k'],
                                outputs=['values', 'indices'])

    def __infer__(self, input_x, k):
        x_shape = list(input_x['shape'])
        ndim = len(x_shape) - 1
        k_v = k['value']
        x_shape[ndim] = k_v
        input_dtype = input_x['dtype']
        validator.check_typename("TopK input_dtype",
                                 input_dtype, (mstype.float16, mstype.float32, mstype.int32))
        if not isinstance(k_v, int):
            raise ValueError('The k must int.', k)
        return {'shape': (x_shape, x_shape),
                'dtype': (input_dtype, mstype.int32),
                'value': None}


class SoftmaxCrossEntropyWithLogits(PrimitiveWithInfer):
    r"""
    Gets the softmax cross-entropy value between logits and labels which shoule be one-hot encoding.

    Note:
        Sets input logits as `X`, input label as `Y`, output as `loss`. Then,

        .. math::
            p_{ij} = softmax(X_{ij}) = \frac{exp(x_i)}{\sum_{j = 0}^{N-1}\exp(x_j)}

        .. math::
            loss_{ij} = -\sum_j{Y_{ij} * ln(p_{ij})}

    Inputs:
        - **logits** (Tensor) - Input logits, with shape :math:`(N, C)`.
        - **labels** (Tensor) - Ground truth labels, with shape :math:`(N, C)`.

    Outputs:
        Tuple of 2 Tensor, the loss shape is `(N,)`, and the dlogits with the same shape as `logits`.
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, logits_shape, labels_shape):
        validator.check_param_equal("SoftmaxCrossEntropyWithLogits logits_shape", logits_shape,
                                    "SoftmaxCrossEntropyWithLogits labels_shape", labels_shape)
        loss_shape = [logits_shape[0]]
        dlogits_shape = logits_shape
        return (loss_shape, dlogits_shape)

    def infer_dtype(self, logits_type, labels_type):
        args = {"SoftmaxCrossEntropyWithLogits logits_type": logits_type,
                "SoftmaxCrossEntropyWithLogits labels_type": labels_type}
        validator.check_type_same(args, (mstype.float16, mstype.float32))
        return (logits_type, logits_type)


class SparseSoftmaxCrossEntropyWithLogits(PrimitiveWithInfer):
    r"""
    Computes the softmax cross-entropy value between logits and sparse encoding labels.

    Note:
        Sets input logits as `X`, input label as `Y`, output as `loss`. Then,

        .. math::
            p_{ij} = softmax(X_{ij}) = \frac{exp(x_i)}{\sum_{j = 0}^{N-1}\exp(x_j)}

        .. math::
            loss_{ij} = \begin{cases} -ln(p_{ij}), &j = y_i \cr -ln(1 - p_{ij}), & j \neq y_i \end{cases}

        .. math::
            loss = \sum_{ij} loss_{ij}

    Args:
        is_grad (bool): If it's true, this operation returns the computed gradient. Default: False.

    Inputs:
        - **logits** (Tensor) - Input logits, with shape :math:`(N, C)`.
        - **labels** (Tensor) - Ground truth labels, with shape :math:`(N)`.

    Outputs:
        Tensor, if `is_grad` is False, the output tensor is the value of loss which is a scalar tensor;
        if `is_grad` is True, the output tensor is the gradient of input with the same shape as `logits`.
    """

    @prim_attr_register
    def __init__(self, is_grad=False):
        self.init_prim_io_names(inputs=['features', 'labels'], outputs=['output'])
        self.is_grad = is_grad
        self.add_prim_attr('sens', 1.0)

    def infer_shape(self, logits_shape, labels_shape):
        validator.check_param_equal("SparseSoftmaxCrossEntropyWithLogits logits_shape", logits_shape[0],
                                    "SparseSoftmaxCrossEntropyWithLogits labels_shape", labels_shape[0])
        loss_shape = []
        if self.is_grad:
            return logits_shape
        return loss_shape

    def infer_dtype(self, logits_type, labels_type):
        validator.check_typename("SparseSoftmaxCrossEntropyWithLogits logits_type",
                                 logits_type, (mstype.float16, mstype.float32))
        validator.check_typename("SparseSoftmaxCrossEntropyWithLogits labels_type",
                                 labels_type, (mstype.int32, mstype.int64))
        return logits_type


class ApplyMomentum(PrimitiveWithInfer):
    """
    Optimizer that implements the Momentum algorithm.

    Refer to the paper `On the importance of initialization and momentum in deep
    learning <https://dl.acm.org/doi/10.5555/3042817.3043064>`_  for more details.

    Args:
        use_locking (bool): Enable a lock to protect the update of variable and accumlation tensors. Default: False.
        use_nesterov (bool): Enable Nesterov momentum. Default: False.
        gradient_scale (float): The scale of the gradient. Default: 1.0.

    Inputs:
        - **variable** (Tensor) - Weights to be updated.
        - **accumulation** (Tensor) - Accumulated gradient value by moment weight.
        - **learning_rate** (float) - Learning rate.
        - **gradient** (Tensor) - Gradients.
        - **momentum** (float) - Momentum.

    Outputs:
        Tensor, parameters to be updated.

    Examples:
        >>> net = ResNet50()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> opt = P.ApplyMomentum(Tensor(np.array([0.001])), Tensor(np.array([0.9])),
                                filter(lambda x: x.requires_grad, net.get_parameters()))
        >>> model = Model(net, loss, opt)
    """
    __mindspore_signature__ = (
        ('variable', sig_rw.RW_WRITE, sig_kind.KIND_POSITIONAL_KEYWORD),
        ('accumulation', sig_rw.RW_WRITE, sig_kind.KIND_POSITIONAL_KEYWORD),
        ('learning_rate', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD),
        ('gradient', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD),
        ('momentum', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD)
    )
    @prim_attr_register
    def __init__(self, use_nesterov=False, use_locking=False, gradient_scale=1.0):
        self.init_prim_io_names(inputs=['variable', 'accumulation', 'learning_rate', 'gradient', 'momentum'],
                                outputs=['output'])

    def infer_shape(self, v_shape, a_shape, l_shape, g_shape, m_shape):
        return v_shape

    def infer_dtype(self, v_dtype, a_dtype, l_dtype, g_dtype, m_dtype):
        if v_dtype != mstype.type_refkey and a_dtype != mstype.type_refkey:
            validator.check_subclass("v_dtype", v_dtype, mstype.tensor)
            validator.check_subclass("a_dtype", a_dtype, mstype.tensor)
            validator.check_typename("v_dtype", v_dtype, [mstype.float16, mstype.float32, mstype.float64])
            validator.check_typename("a_dtype", a_dtype, [mstype.float16, mstype.float32, mstype.float64])
        validator.check_typename("l_dtype", l_dtype, [mstype.float16, mstype.float32, mstype.float64])
        validator.check_typename("g_dtype", g_dtype, [mstype.float16, mstype.float32, mstype.float64])
        validator.check_typename("m_dtype", m_dtype, [mstype.float16, mstype.float32, mstype.float64])
        return g_dtype


class SmoothL1Loss(PrimitiveWithInfer):
    r"""
    Computes smooth L1 loss, a robust L1 loss.

    SmoothL1Loss is a Loss similar to MSELoss but less sensitive to outliers as described in the
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_ by Ross Girshick.

    Note:
        Sets input prediction as `X`, input target as `Y`, output as `loss`. Then,

        .. math::
            \text{SmoothL1Loss} = \begin{cases}0.5x^{2}, &if \left |x \right |\leq \text{sigma} \cr
            \left |x \right|-0.5, &\text{otherwise}\end{cases}

    Args:
        sigma (float): A parameter used to control the point where the function will change from
            quadratic to linear. Default: 1.0.

    Inputs:
        - **prediction** (Tensor) - Predict data.
        - **target** (Tensor) - Ground truth data, with the same type and shape as `prediction`.

    Outputs:
        Tensor, with the same type and shape as `prediction`.
    """

    @prim_attr_register
    def __init__(self, sigma=1.0):
        validator.check_type('sigma', sigma, [float])
        validator.check('sigma', sigma, '', 0, Rel.GT)
        self.init_prim_io_names(inputs=['prediction', 'target'], outputs=['output'])

    def infer_shape(self, prediction, target):
        validator.check_param_equal('prediction shape', prediction, 'target shape', target)
        return prediction

    def infer_dtype(self, prediction, target):
        args = {"prediction": prediction, "target": target}
        validator.check_type_same(args, (mstype.float16, mstype.float32))
        return prediction


class SGD(PrimitiveWithInfer):
    """
    Computes stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from On the importance of
    initialization and momentum in deep learning.

    Note:
        For details, please refer to `nn.SGD` source code.

    Args:
        dampening (float): The dampening for momentum. Default: 0.0.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0.
        nesterov (bool): Enable Nesterov momentum. Default: False.

    Inputs:
        - **parameters** (Tensor) - Parameters to be updated.
        - **gradient** (Tensor) - Gradients.
        - **learning_rate** (Tensor) - Learning rate. e.g. Tensor(0.1, mindspore.float32).
        - **accum** (Tensor) - Accum(velocity) to be updated.
        - **momentum** (Tensor) - Momentum. e.g. Tensor(0.1, mindspore.float32).
        - **stat** (Tensor) - States to be updated with the same shape as gradient.

    Outputs:
        Tensor, parameters to be updated.
    """

    @prim_attr_register
    def __init__(self, dampening=0.0, weight_decay=0.0, nesterov=False):
        self.init_prim_io_names(inputs=['parameters', 'gradient', 'learning_rate', 'accum', 'momentum', 'stat'],
                                outputs=['output'])

    def infer_shape(self, parameters_shape, gradient_shape, learning_rate_shape,
                    accum_shape, momentum_shape, stat_shape):
        validator.check(f'parameters shape {parameters_shape}', len(parameters_shape), '', 0, Rel.GT)
        validator.check(f'gradient shape {gradient_shape}', len(gradient_shape), '', 0, Rel.GE)
        validator.check(f'learning rate shape {learning_rate_shape}', len(learning_rate_shape), '', 0, Rel.GE)
        validator.check(f'accumulation shape {accum_shape}', len(accum_shape), '', 0, Rel.GT)
        validator.check(f'momentum shape {momentum_shape}', len(momentum_shape), '', 0, Rel.GE)
        validator.check(f'stat shape {stat_shape}', len(stat_shape), '', 0, Rel.GE)
        validator.check("gradient shape", gradient_shape, "stat shape", stat_shape)
        return parameters_shape

    def infer_dtype(self, parameters_dtype, gradient_dtype, learning_rate_dtype,
                    accum_dtype, momentum_dtype, stat_dtype):
        validator.check_typename("parameters_dtype", parameters_dtype, [mstype.float16, mstype.float32])
        validator.check_typename("gradient_dtype", gradient_dtype, [mstype.float16, mstype.float32])
        validator.check_typename("learning_rate_dtype", learning_rate_dtype, [mstype.float16, mstype.float32])
        validator.check_typename("accum_dtype", accum_dtype, [mstype.float16, mstype.float32])
        validator.check_typename("momentum_dtype", momentum_dtype, [mstype.float16, mstype.float32])
        validator.check_typename("stat_dtype", stat_dtype, [mstype.float16, mstype.float32])
        return parameters_dtype

class ApplyRMSProp(PrimitiveWithInfer):
    """
    Optimizer that implements the Root Mean Square prop(RMSProp) algorithm.
    Please refer to the usage in source code of `nn.RMSProp`.

    Note:
        Update `var` according to the RMSProp algorithm.

        ..  math::
            s_{t} = \\rho s_{t-1} + (1 - \\rho)(\\nabla Q_{i}(w))^2

        ..  math::
            m_{t} = \\beta m_{t-1} + \\frac{\\eta} {\\sqrt{s_{t} + \\epsilon}} \\nabla Q_{i}(w)

        ..  math::
            w = w - m_{t}

        where, :math:`w` represents `var`, which will be updated.
        :math:`s_{t}` represents `mean_square`, :math:`s_{t-1}` is the last momentent of :math:`s_{t}`,
        :math:`m_{t}` represents `moment`, :math:`m_{t-1}` is the last momentent of :math:`m_{t}`.
        :math:`\\rho` represents `decay`. :math:`\\beta` is the momentum term, represents `momentum`.
        :math:`\\epsilon` is a smoothing term to avoid division by zero, represents `epsilon`.
        :math:`\\eta` represents `learning_rate`. :math:`\\nabla Q_{i}(w)` represents `grad`.

    Args:
        use_locking (bool): Enable a lock to protect the update of variable tensors. Default: False.

    Inputs:
        - **var** (Tensor) - Weights to be update.
        - **mean_square** (Tensor) - Mean square gradients, must have the same type as `var`.
        - **moment** (Tensor) - Delta of `var`, must have the same type as `var`.
        - **grad** (Tensor) - Gradients, must have the same type as `var`.
        - **learning_rate** (Union[Number, Tensor]) - Learning rate.
        - **decay** (float) - Decay rate.
        - **momentum** (float) - Momentum.
        - **epsilon** (float) - Ridge term.

    Outputs:
        Tensor, parameters to be update.
    """

    @prim_attr_register
    def __init__(self, use_locking=False):
        self.use_locking = validator.check_type("use_locking", use_locking, [bool])

    def infer_shape(self, var_shape, mean_square_shape, moment_shape, grad_shape, learning_rate_shape, decay_shape,
                    momentum_shape, epsilon_shape):
        validator.check_param_equal("var_shape", var_shape, "mean_square_shape", mean_square_shape)
        validator.check_param_equal("var_shape", var_shape, "moment_shape", moment_shape)
        validator.check_param_equal("var_shape", var_shape, "grad_shape", grad_shape)
        return var_shape

    def infer_dtype(self, var_dtype, mean_square_dtype, moment_dtype, grad_dtype, learning_rate_dtype, decay_dtype,
                    momentum_dtype, epsilon_dtype):
        validator.check_subclass("var_dtype", var_dtype, mstype.tensor)
        validator.check_subclass("mean_square_dtype", mean_square_dtype, mstype.tensor)
        validator.check_subclass("moment_dtype", moment_dtype, mstype.tensor)
        validator.check_subclass("grad_dtype", moment_dtype, mstype.tensor)
        args = {"var_dtype": var_dtype, "mean_square_dtype": mean_square_dtype, "moment_dtype": moment_dtype,
                "grad_dtype": grad_dtype}
        validator.check_type_same(args, mstype.number_type)

        args = {"learning_rate_dtype": learning_rate_dtype, "decay_dtype": decay_dtype,
                'momentum_dtype': momentum_dtype, "epsilon_dtype": epsilon_dtype}
        validator.check_type_same(args, [mstype.float16, mstype.float32])
        return var_dtype


class ApplyCenteredRMSProp(PrimitiveWithInfer):
    """
    Optimizer that implements the centered RMSProp algorithm.
    Please refer to the usage in source code of `nn.RMSProp`.

    Note:
        Update `var` according to the centered RMSProp algorithm.

        ..  math::
            g_{t} = \\rho g_{t-1} + (1 - \\rho)\\nabla Q_{i}(w)

        ..  math::
            s_{t} = \\rho s_{t-1} + (1 - \\rho)(\\nabla Q_{i}(w))^2

        ..  math::
            m_{t} = \\beta m_{t-1} + \\frac{\\eta} {\\sqrt{s_{t} - g_{t}^2 + \\epsilon}} \\nabla Q_{i}(w)

        ..  math::
            w = w - m_{t}

        where, :math:`w` represents `var`, which will be updated.
        :math:`g_{t}` represents `mean_gradient`, :math:`g_{t-1}` is the last momentent of :math:`g_{t}`.
        :math:`s_{t}` represents `mean_square`, :math:`s_{t-1}` is the last momentent of :math:`s_{t}`,
        :math:`m_{t}` represents `moment`, :math:`m_{t-1}` is the last momentent of :math:`m_{t}`.
        :math:`\\rho` represents `decay`. :math:`\\beta` is the momentum term, represents `momentum`.
        :math:`\\epsilon` is a smoothing term to avoid division by zero, represents `epsilon`.
        :math:`\\eta` represents `learning_rate`. :math:`\\nabla Q_{i}(w)` represents `grad`.

    Args:
        use_locking (bool): Enable a lock to protect the update of variable tensors. Default: False.

    Inputs:
        - **var** (Tensor) - Weights to be update.
        - **mean_gradient** (Tensor) - Mean gradients, must have the same type as `var`.
        - **mean_square** (Tensor) - Mean square gradients, must have the same type as `var`.
        - **moment** (Tensor) - Delta of `var`, must have the same type as `var`.
        - **grad** (Tensor) - Gradients, must have the same type as `var`.
        - **learning_rate** (Union[Number, Tensor]) - Learning rate.
        - **decay** (float) - Decay rate.
        - **momentum** (float) - Momentum.
        - **epsilon** (float) - Ridge term.

    Outputs:
        Tensor, parameters to be update.
    """

    @prim_attr_register
    def __init__(self, use_locking=False):
        self.use_locking = validator.check_type("use_locking", use_locking, [bool])

    def infer_shape(self, var_shape, mean_gradient_shape, mean_square_shape, moment_shape, grad_shape,
                    learning_rate_shape, decay_shape, momentum_shape, epsilon_shape):
        validator.check_param_equal("var_shape", var_shape, "mean_gradient_shape", mean_gradient_shape)
        validator.check_param_equal("var_shape", var_shape, "mean_square_shape", mean_square_shape)
        validator.check_param_equal("var_shape", var_shape, "moment_shape", moment_shape)
        validator.check_param_equal("var_shape", var_shape, "grad_shape", grad_shape)
        return var_shape

    def infer_dtype(self, var_dtype, mean_gradient_dtype, mean_square_dtype, moment_dtype, grad_dtype,
                    learning_rate_dtype, rho_dtype, momentum_dtype, epsilon_dtype):
        validator.check_subclass("var_dtype", var_dtype, mstype.tensor)
        validator.check_subclass("mean_gradient_dtype", mean_gradient_dtype, mstype.tensor)
        validator.check_subclass("mean_square_dtype", mean_square_dtype, mstype.tensor)
        validator.check_subclass("moment_dtype", moment_dtype, mstype.tensor)
        validator.check_subclass("grad_dtype", moment_dtype, mstype.tensor)
        args = {"var_dtype": var_dtype, "mean_gradient_dtype": mean_gradient_dtype,
                "mean_square_dtype": mean_square_dtype, "moment_dtype": moment_dtype, "grad_dtype": grad_dtype}
        validator.check_type_same(args, mstype.number_type)

        args = {"learning_rate_dtype": learning_rate_dtype, "rho_dtype": rho_dtype, 'momentum_dtype': momentum_dtype,
                "epsilon_dtype": epsilon_dtype}
        validator.check_type_same(args, [mstype.float16, mstype.float32])
        return var_dtype


class LayerNorm(Primitive):
    r"""
    Applies the Layer Normalization to the input tensor.

    This operator will normalize the input tensor on given axis. LayerNorm is described in the paper
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`_.

    .. math::
        y = \frac{x - mean]}{\sqrt{variance + \epsilon}} * \gamma + \beta

    where :math:`\gamma` is scale, :math:`\beta` is bias, :math:`\epsilon` is epsilon.

    Args:
        begin_norm_axis (int): The begin axis of the `input_x` to apply LayerNorm,
            the value should be in [-1, rank(input)). Default: 1.
        begin_params_axis (int): The begin axis of the parameter input (`gamma`, `beta`) to
            apply LayerNorm, the value should be in [-1, rank(input)). Default: 1.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
          The input of LayerNorm.
        - **gamma** (Tensor) - Tensor of shape :math:`(P_0, \ldots, P_\text{begin_params_axis})`.
          The learnable parameter `gamma` as the scale on norm.
        - **beta** (Tensor) - Tensor of shape :math:`(P_0, \ldots, P_\text{begin_params_axis})`.
          The learnable parameter `beta` as the scale on norm.

    Outputs:
        tuple[Tensor], tuple of 3 tensors, the normalized input and the updated parameters.

        - **output_x** (Tensor) - The normalized input, has the same type and shape as the `input_x`.
          The shape is :math:`(N, C)`.
        - **updated_gamma** (Tensor) - Tensor of shape :math:`(C,)`.
        - **updated_beta** (Tensor) - Tensor of shape :math:`(C,)`.
    """

    @prim_attr_register
    def __init__(self, begin_norm_axis=1, begin_params_axis=1):
        validator.check_type('begin_norm_axis', begin_norm_axis, [int])
        validator.check_type('begin_params_axis', begin_params_axis, [int])


class L2Normalize(PrimitiveWithInfer):
    r"""
    L2 normalization Operator.

    This operator will normalizes the input using the given axis. The function is shown as follows:

    .. math::
        \text{output} = \frac{x}{\sqrt{\text{max}(\text{sum} (\text{input_x}^2), \epsilon)}},

    where :math:`\epsilon` is epsilon.

    Args:
        axis (int): The begin axis for the input to apply L2 normalize. Default: 0.
        epsilon (float): A small value added for numerical stability. Default: 1e-4.

    Inputs:
        - **input_x** (Tensor) - Input to compute the normalization.

    Outputs:
        Tensor, with the same type and shape as the input.
    """

    @prim_attr_register
    def __init__(self, axis=0, epsilon=1e-4):
        validator.check_type('axis', axis, [int])
        validator.check_type('epsilon', epsilon, [int, float])

    def infer_shape(self, input_x):
        dim = len(input_x)
        validator.check_int_range('axis value', self.axis, -dim, dim, Rel.INC_LEFT)
        return input_x

    def infer_dtype(self, input_x):
        validator.check_subclass("x", input_x, mstype.tensor)
        return input_x


class DropoutGenMask(Primitive):
    """
    Generates the mask value for the input shape.

    Args:
        Seed0 (int): Seed0 value for random generating. Default: 0.
        Seed1 (int): Seed1 value for random generating. Default: 0.

    Inputs:
        - **shape** (tuple[int]) - The shape of target mask.
        - **keep_prob** (Tensor) - The keep rate, between 0 and 1, e.g. keep_prob = 0.9,
          means dropping out 10% of input units.

    Outputs:
        Tensor, the value of generated mask for input shape.

    Examples:
        >>> dropout_gen_mask = P.DropoutGenMask()
        >>> shape = (20, 16, 50)
        >>> keep_prob = Tensor(0.5, mindspore.float32)
        >>> mask = dropout_gen_mask(shape, keep_prob)
    """

    @prim_attr_register
    def __init__(self, Seed0=0, Seed1=0):
        self.init_prim_io_names(inputs=['shape', 'keep_prob'], outputs=['output'])
        validator.check_type("Seed0", Seed0, [int])
        validator.check_type("Seed1", Seed1, [int])


class DropoutDoMask(PrimitiveWithInfer):
    """
    Applies dropout mask on the input tensor.

    Take the mask output of DropoutGenMask as input, and apply dropout on the input.

    Inputs:
        - **input_x** (Tensor) - The input tensor.
        - **mask** (Tensor) - The mask to be applied on `input_x`, which is the output of `DropoutGenMask`. And the
          shape of `input_x` must be same as the value of `DropoutGenMask`'s input `shape`. If input wrong `mask`,
          the output of `DropoutDoMask` are unpredictable.
        - **keep_prob** (Tensor) - The keep rate, between 0 and 1, e.g. keep_prob = 0.9,
          means dropping out 10% of input units. The value of `keep_prob` is same as the input `keep_prob` of
          `DropoutGenMask`.

    Outputs:
        Tensor, the value that applied dropout on.

    Examples:
        >>> x = Tensor(np.ones([20, 16, 50]), mindspore.float32)
        >>> shape = (20, 16, 50)
        >>> keep_prob = Tensor(0.5, mindspore.float32)
        >>> dropout_gen_mask = P.DropoutGenMask()
        >>> dropout_do_mask = P.DropoutDoMask()
        >>> mask = dropout_gen_mask(shape, keep_prob)
        >>> output = dropout_do_mask(x, mask, keep_prob)
        >>> assert output.shape() == (20, 16, 50)
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __infer__(self, input_x, mask, keep_prob):
        input_x_shape = input_x['shape']
        mask_shape = mask['shape']
        keep_prob_shape = keep_prob['shape']
        validator.check("keep_prob's dim", len(keep_prob_shape), '0(scalar)', 0)
        size_x = reduce(lambda x, y: x * y, input_x_shape)
        if len(mask_shape) != 1:
            raise ValueError("DropoutDoMask mask shape should be 1-dimension.")
        size_y = mask_shape[0] * 8
        if size_x > size_y:
            raise ValueError(f"DropoutDoMask y mask do not math input input_x shape:"
                             "{input_x_shape}, mask shape: {mask_shape}.")

        validator.check_typename("input_x type", input_x['dtype'], [mstype.float32, mstype.float16, mstype.int32])
        validator.check_typename("input_mask type", mask['dtype'], [mstype.uint8])

        keep_prob_v = keep_prob['value']
        if keep_prob_v is not None:
            validator.check_const_input('keep_prob', keep_prob_v)
            validator.check_number_range('keep_prob', keep_prob_v.asnumpy(), 0, 1, Rel.INC_BOTH)

        out = {'shape': input_x_shape,
               'dtype': input_x['dtype'],
               'value': None}
        return out


class ResizeBilinear(PrimitiveWithInfer):
    r"""
    Resizes the image to certain size using bilinear interpolation.

    The resizing only affects the lower two dimensions which represent the height and width. The input images
    can be represented by different data types, but the data types of output images are always float32.

    Args:
        size (tuple[int]): A tuple of 2 int elements `(new_height, new_width)`, the new size for the images.
        align_corners (bool): If it's true, rescale input by `(new_height - 1) / (height - 1)`,
                       which exactly aligns the 4 corners of images and resized images. If it's false,
                       rescale by `new_height / height`. Default: False.

    Inputs:
        - **input** (Tensor) - Image to be resized. Tensor of shape `(N_i, ..., N_n, height, width)`.

    Outputs:
        Tensor, resized image. Tensor of shape `(N_i, ..., N_n, new_height, new_width)` in `float32`.

    Examples:
        >>> tensor = Tensor([[[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]], mindspore.int32)
        >>> resize_bilinear = P.ResizeBilinear((5, 5))
        >>> result = resize_bilinear(tensor)
        >>> assert result.shape() == (5, 5)
    """

    @prim_attr_register
    def __init__(self, size, align_corners=False):
        pass

    def infer_shape(self, input_shape):
        input_shape = list(input_shape)
        batch, channel, _, _ = input_shape
        out_shape = [batch, channel]
        for i in self.size:
            out_shape.append(int(i))
        return out_shape

    def infer_dtype(self, input_dtype):
        return mstype.tensor_type(mstype.float32)


class OneHot(PrimitiveWithInfer):
    r"""
    Computes a one-hot tensor.

    Makes a new tensor, whose locations represented by indices in `indices` take value `on_value`, while all
    other locations take value `off_value`.

    Note:
        If the input indices is rank `N`, the output will have rank `N+1`. The new axis is created at dimension `axis`.

    Args:
        axis (int): Position to insert the value. e.g. If `indices` shape is [n, c], and `axis` is `-1` the output shape
            will be [n, c, depth], If `axis` is `0` the output shape will be [depth, n, c]. Default: -1.

    Inputs:
        - **indices** (Tensor) - A tensor of indices. Tensor of shape :math:`(X_0, \ldots, X_n)`.
        - **depth** (int) - A scalar defining the depth of the one hot dimension.
        - **on_value** (Tensor) - A value to fill in output when `indices[j] = i`.
        - **off_value** (Tensor) - A value to fill in output when `indices[j] != i`.

    Outputs:
        Tensor, one_hot tensor. Tensor of shape :math:`(X_0, \ldots, X_{axis}, \text{depth} ,X_{axis+1}, \ldots, X_n)`.

    Examples:
        >>> indices = Tensor(np.array([0, 1, 2]), mindspore.int32)
        >>> depth, on_value, off_value = 3, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
        >>> onehot = P.OneHot()
        >>> result = onehot(indices, depth, on_value, off_value)
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """

    @prim_attr_register
    def __init__(self, axis=-1):
        self.init_prim_io_names(inputs=['indices', 'depth', 'on_value', 'off_value'], outputs=['output'])
        validator.check_type("axis", axis, [int])

    def __infer__(self, indices, depth, on_value, off_value):
        # check type
        validator.check_subclass("indices", indices['dtype'], mstype.tensor)
        validator.check_typename("indices", indices['dtype'], (mstype.int32,))
        validator.check_typename("depth", depth['dtype'], mstype.int_type)
        validator.check_subclass("on_value", on_value['dtype'], mstype.tensor)
        validator.check_subclass("off_value", off_value['dtype'], mstype.tensor)
        args = {"on_value dtype": on_value['dtype'], "off_value dtype": off_value['dtype']}
        validator.check_type_same(args, (mstype.float16, mstype.float32))

        # check shape
        indices_shp = indices['shape']
        validator.check_int_range("axis", self.axis, -1, len(indices_shp), Rel.INC_BOTH)
        depth_val = depth['value']
        validator.check_integer("depth", depth_val, 0, Rel.GE)
        # create new dimension at end if self.axis is -1
        indices_shp.insert(self.axis, depth_val) if self.axis >= 0 else indices_shp.append(depth_val)

        return {'shape': indices_shp,
                'dtype': on_value['dtype'],
                'value': None}


class Gelu(PrimitiveWithInfer):
    r"""
    Gaussian Error Linear Units activation function.

    GeLU is described in the paper `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.
    And also please refer to `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
    <https://arxiv.org/abs/1810.04805>`_.

    Defined as follows:

    .. math::
        \text{output} = 0.5 * x * (1 + erf(x / \sqrt{2})),

    where :math:`erf` is the "Gauss error function" .

    Inputs:
        - **input_x** (Tensor) - Input to compute the Gelu.

    Outputs:
        Tensor, with the same type and shape as input.

    Examples:
        >>> tensor = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> gelu = P.Gelu()
        >>> result = gelu(tensor)
    """

    @prim_attr_register
    def __init__(self):
        """init GeLU"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_subclass("input_x", input_x, mstype.tensor)
        validator.check_typename("input_x", input_x, (mstype.float16, mstype.float32))
        return input_x


class GetNext(PrimitiveWithInfer):
    """
    Returns the next element in the dataset queue.

    Note:
        GetNext op needs to be associated with network and also depends on the init_dataset interface,
        it can't be used directly as a single op.
        For details, please refer to `nn.cell_wrapper.DataWrapper` source code.

    Args:
        types (list[:class:`mindspore.dtype`]): The type of the outputs.
        shapes (list[tuple[int]]): The dimensionality of the outputs.
        output_num (int): The output number, length of `types` and `shapes`.
        shared_name (str): The queue name of `init_dataset` interface.

    Inputs:
        No inputs.

    Outputs:
        tuple[Tensor], the output of Dataset. The shape is described in `shapes`
        and the type is described is `types`.

    Examples:
        >>> get_next = P.GetNext([mindspore.float32, mindspore.int32], [[32, 1, 28, 28], [10]], 'shared_name')
        >>> feature, label = get_next()
    """

    @prim_attr_register
    def __init__(self, types, shapes, output_num, shared_name):
        validator.check_type("types", types, [list, tuple])
        validator.check_type("shapes", shapes, [list, tuple])
        validator.check("types length", len(types), "shapes length", len(shapes))
        validator.check_type("output_num", output_num, [int])

    def infer_shape(self):
        return tuple(self.shapes)

    def infer_dtype(self):
        return tuple(self.types)


class PReLU(PrimitiveWithInfer):
    r"""
    Parametric Rectified Linear Unit activation function.

    PReLU is described in the paper `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_. Defined as follows:

    .. math::
        prelu(x_i)= \max(0, x_i) + \min(0, w * x_i),

    where :math:`x_i` is an element of an channel of the input.

    Inputs:
        - **input_x** (Tensor) - Float tensor, representing the output of the preview layer.
        - **weight** (Tensor) -  Float Tensor, w > 0, there is only two shapes are legitimate,
          1 or the number of channels at input.

    Outputs:
        Tensor, with the same type as `input_x`.

    Detailed information, please refer to `nn.PReLU`.
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, input_x_shape, weight_shape):
        input_x_dim = len(input_x_shape)
        weight_dim = len(weight_shape)

        if weight_dim != 1:
            raise ValueError(f'weight_dim must be 1, while weight_dim is {weight_dim}.')

        if input_x_dim == 1 and weight_shape[0] != 1:
            raise ValueError(f'when input_x_dim is 1, weight_shape[0] must be 1, '
                             f'while weight_shape[0] is {weight_shape[0]}.')

        if input_x_dim != 1 and weight_shape[0] != input_x_shape[1] and weight_shape[0] != 1:
            raise ValueError(f'channel of input_x and weight must be matched,'
                             f' while channel of input_x is {input_x_shape[1]},'
                             f' weight_shape[0] is {weight_shape[0]}.')

        return input_x_shape

    def infer_dtype(self, input_x_dtype, weight_dtype):
        validator.check_subclass("input_x_dtype", input_x_dtype, mstype.tensor)
        validator.check_subclass("weight_dtype", weight_dtype, mstype.tensor)
        validator.check_typename("input_x_dtype", input_x_dtype, (mstype.float16, mstype.float32))
        validator.check_typename("weight_dtype", weight_dtype, (mstype.float16, mstype.float32))
        return input_x_dtype


class LSTM(PrimitiveWithInfer):
    """
    Performs the long short term memory(LSTM) on the input.

    Detailed information, please refer to `nn.LSTM`.
    """

    @prim_attr_register
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        self.input_size = check_int_positive(input_size)
        self.hidden_size = check_int_positive(hidden_size)
        self.num_layers = check_int_positive(num_layers)
        self.has_bias = check_bool(has_bias)
        self.bidirectional = check_bool(bidirectional)
        self.dropout = validator.check_type("dropout", dropout, [float])
        self.dropout = validator.check_number_range('dropout', dropout, 0, 1, Rel.INC_BOTH)

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

    def infer_shape(self, x_shape, h_shape, c_shape, w_shape):
        # (batch, seq, feature)
        validator.check_integer("x_shape", len(x_shape), 3, Rel.EQ)

        # h and c should be same shape
        validator.check_integer("h_shape", len(h_shape), 3, Rel.EQ)
        validator.check_integer("h_shape", len(h_shape), len(c_shape), Rel.EQ)
        validator.check_integer("h_shape", h_shape[0], c_shape[0], Rel.EQ)
        validator.check_integer("h_shape", h_shape[1], c_shape[1], Rel.EQ)
        validator.check_integer("h_shape", h_shape[2], c_shape[2], Rel.EQ)

        # (num_layers * num_directions, batch, hidden_size)
        validator.check_integer("h[0]", h_shape[0], self.num_layers * self.num_directions, Rel.EQ)
        validator.check_integer("h[1]", h_shape[1], x_shape[1], Rel.EQ)
        validator.check_integer("h[2]", h_shape[2], self.hidden_size, Rel.EQ)

        y_shape = (x_shape[0], x_shape[1], self.hidden_size * self.num_directions)

        # set arbitrary shape for reserved space
        reserved_shape = (1, 1)
        state_shape = (1, 1)
        return (y_shape, h_shape, c_shape, reserved_shape, state_shape)

    def infer_dtype(self, x_dtype, h_dtype, c_dtype, w_dtype):
        validator.check_typename("x_dtype", x_dtype, (mstype.float32, mstype.float16))
        validator.check_typename("h_dtype", h_dtype, (mstype.float32, mstype.float16))
        validator.check_typename("c_dtype", c_dtype, (mstype.float32, mstype.float16))
        validator.check_typename("w_dtype", w_dtype, (mstype.float32, mstype.float16))
        validator.check_typename("datatype", x_dtype, (h_dtype.element_type(),))
        validator.check_typename("datatype", x_dtype, (c_dtype.element_type(),))
        validator.check_typename("datatype", x_dtype, (w_dtype.element_type(),))
        return (x_dtype, x_dtype, x_dtype, x_dtype, x_dtype)


class SigmoidCrossEntropyWithLogits(PrimitiveWithInfer):
    r"""
    Uses the given logits to compute sigmoid cross entropy.

    Note:
        Sets input logits as `X`, input label as `Y`, output as `loss`. Then,

        .. math::
            p_{ij} = sigmoid(X_{ij}) = \frac{1}{1 + e^{-X_{ij}}}

        .. math::
            loss_{ij} = -[Y_{ij} * ln(p_{ij}) + (1 - Y_{ij})ln(1 - p_{ij})]

    Inputs:
        - **logits** (Tensor) - Input logits.
        - **label** (Tensor) - Ground truth label.

    Outputs:
        Tensor, with the same shape and type as input `logits`.
    """

    @prim_attr_register
    def __init__(self):
        """Init SigmoidCrossEntropyWithLogits"""
        self.init_prim_io_names(inputs=['predict', 'target'], outputs=['loss'])

    def infer_shape(self, x_shape, y_shape):
        validator.check_param_equal("x_shape", x_shape, "y_shape", y_shape)
        return x_shape

    def infer_dtype(self, x_dtype, y_dtype):
        args = {"x_dtype": x_dtype, "y_dtype": y_dtype}
        validator.check_type_same(args, mstype.number_type)
        return x_dtype


class Pad(PrimitiveWithInfer):
    """
    Pads input tensor according to the paddings.

    Args:
        paddings (tuple): The shape of parameter `paddings` is (N, 2). N is the rank of input data. All elements of
            paddings are int type. For `D` th dimension of input, paddings[D, 0] indicates how many sizes to be
            extended ahead of the `D` th dimension of the input tensor, and paddings[D, 1] indicates how many sizes to
            be extended behind of the `D` th dimension of the input tensor.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the tensor after padding.

    Examples:
        >>> input_tensor = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> pad_op = P.Pad(((1, 2), (2, 1)))
        >>> output_tensor = pad_op(input_tensor)
        >>> assert output_tensor == Tensor(np.array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
        >>>                                          [ 0. ,  0. , -0.1,  0.3,  3.6,  0. ],
        >>>                                          [ 0. ,  0. ,  0.4,  0.5, -3.2,  0. ],
        >>>                                          [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
        >>>                                          [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ]]), mindspore.float32)
    """

    @prim_attr_register
    def __init__(self, paddings):
        """Init Pad"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        if not isinstance(paddings, tuple):
            raise TypeError('Paddings must be tuple type.')
        for item in paddings:
            if len(item) != 2:
                raise ValueError('The shape of paddings must be (n, 2).')
        self.paddings = paddings

    def infer_shape(self, x):
        paddings = np.array(self.paddings)
        validator.check_integer('paddings.shape', paddings.size, len(x) * 2, Rel.EQ)
        if not np.all(paddings >= 0):
            raise ValueError('All elements of paddings must be >= 0.')
        y_shape = ()
        for i in range(int(paddings.size / 2)):
            y_shape += ((x[i] + paddings[i, 0] + paddings[i, 1]),)
        return y_shape

    def infer_dtype(self, x):
        validator.check_subclass("input_x", x, mstype.tensor)
        return x


class MirrorPad(PrimitiveWithInfer):
    """
    Pads the input tensor according to the paddings and mode.

    Args:
        mode (string): Specifies padding mode. The optional values are "REFLECT", "SYMMETRIC".
            Default: "REFLECT".

    Inputs:
        - **input_x** (Tensor) - The input tensor.
        - **paddings** (Tensor) - The paddings tensor. The value of `paddings` is a matrix(list),
            and its shape is (N, 2). N is the rank of input data. All elements of paddings
            are int type. For `D` th dimension of input, paddings[D, 0] indicates how many sizes to be
            extended ahead of the `D` th dimension of the input tensor, and paddings[D, 1] indicates
            how many sizes to be extended behind of the `D` th dimension of the input tensor.

    Outputs:
        Tensor, the tensor after padding.

        - If 'mode` is "REFLECT", it uses a way of symmetrical copying throught the axis of symmetry to fill in,
          symmetry. If the `input_x` is [[1,2,3],[4,5,6],[7,8,9]] and `paddings` is [[1,1],[2,2]], then the
          Outputs is [[6,5,4,5,6,5,4],[3,2,1,2,3,2,1],[6,5,4,5,6,5,4],[9,8,7,8,9,8,7],[6,5,4,5,6,5,4]].
        - If 'mode' is "SYMMETRIC", the filling method is similar to the "REFLECT". It is also copied
          according to the symmetry axis, except that it includes the symmetry axis. If the `input_x`
          is [[1,2,3],[4,5,6],[7,8,9]] and `paddings` is [[1,1],[2,2]], then the Outputs is
          [[2,1,1,2,3,3,2],[2,1,1,2,3,3,2],[5,4,4,5,6,6,5],[8,7,7,8,9,9,8],[8,7,7,8,9,9,8]].

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore.ops import operations as P
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.pad = P.MirrorPad(mode="REFLECT")
        >>>     def construct(self, x, paddings):
        >>>         return self.pad(x, paddings)
        >>> x = np.random.random(size=(2, 3)).astype(np.float32)
        >>> paddings = Tensor([[1,1],[2,2]])
        >>> pad = Net()
        >>> ms_output = pad(Tensor(x), paddings)
    """

    @prim_attr_register
    def __init__(self, mode='REFLECT'):
        """Init Pad"""
        validator.check_string('mode', mode, ['REFLECT', 'SYMMETRIC'])
        self.mode = mode

    def __infer__(self, input_x, paddings):
        validator.check_subclass("input_x", input_x['dtype'], mstype.tensor)
        validator.check_subclass("paddings", paddings['dtype'], mstype.tensor)
        x_shape = list(input_x['shape'])
        paddings_value = paddings['value'].asnumpy()
        paddings_size = paddings_value.size
        validator.check_integer('paddings.shape', paddings_size, len(x_shape) * 2, Rel.EQ)
        if not np.all(paddings_size >= 0):
            raise ValueError('All elements of paddings must be >= 0.')
        y_shape = ()
        for i in range(0, int(paddings_size / 2)):
            y_shape += ((x_shape[i] + paddings_value[i, 0] + paddings_value[i, 1]),)

        return {'shape': y_shape,
                'dtype': input_x['dtype'],
                'value': None}


class ROIAlign(PrimitiveWithInfer):
    """
    Computes Region of Interest (RoI) Align operator.

    The operator computes the value of each sampling point by bilinear interpolation from the nearby grid points on the
    feature map. No quantization is performed on any coordinates involved in the RoI, its bins, or the sampling
    points. The details of (RoI) Align operator are described in `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_.

    Args:
        pooled_height (int): The output features' height.
        pooled_width (int): The output features' width.
        spatial_scale (float): A scaling factor that maps the raw image coordinates to the input
            feature map coordinates. Suppose the height of a RoI is `ori_h` in the raw image and `fea_h` in the
            input feature map, the `spatial_scale` should be `fea_h / ori_h`.
        sample_num (int): Number of sampling points. Default: 2.

    Inputs:
        - **features** (Tensor) - The input features, whose shape should be `(N, C, H, W)`.
        - **rois** (Tensor) - The shape is `(rois_n, 5)`. `rois_n` represents the number of RoI. The size of
          the second dimension should be `5` and the `5` colunms are
          `(image_index, top_left_x, top_left_y, bottom_right_x, bottom_right_y)`. `image_index` represents the
          index of image. `top_left_x` and `top_left_y` represent the `x, y` coordinates of the top left corner
          of corresponding RoI, respectively. `bottom_right_x` and `bottom_right_y` represent the `x, y`
          coordinates of the bottom right corner of corresponding RoI, respectively.

    Outputs:
        Tensor, the shape is `(rois_n, C, pooled_height, pooled_width)`.

    Examples:
        >>> input_tensor = Tensor(np.array([[[[1., 2.], [3., 4.]]]]), mindspore.float32)
        >>> rois = Tensor(np.array([[0, 0.2, 0.3, 0.2, 0.3]]), mindspore.float32)
        >>> roi_align = P.ROIAlign(1, 1, 0.5, 2)
        >>> output_tensor = roi_align(input_tensor, rois)
        >>> assert output_tensor == Tensor(np.array([[[[2.15]]]]), mindspore.float32)
    """

    @prim_attr_register
    def __init__(self, pooled_height, pooled_width, spatial_scale, sample_num=2):
        """init ROIAlign"""
        validator.check_type("pooled_height", pooled_height, [int])
        validator.check_type("pooled_width", pooled_width, [int])
        validator.check_type("spatial_scale", spatial_scale, [float])
        validator.check_type("sample_num", sample_num, [int])
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.spatial_scale = spatial_scale
        self.sample_num = sample_num

    def infer_shape(self, inputs_shape, rois_shape):
        return [rois_shape[0], inputs_shape[1], self.pooled_height, self.pooled_width]

    def infer_dtype(self, inputs_type, rois_type):
        return inputs_type


class Adam(PrimitiveWithInfer):
    r"""
    Updates gradients by Adaptive Moment Estimation (Adam) algorithm.

    The Adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            l = \alpha * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
            w = w - l * \frac{m}{\sqrt{v} + \epsilon}
        \end{array}

    :math:`m` represents the 1st moment vector, :math:`v` represents the 2nd moment vector, :math:`g` represents
    `gradient`, :math:`l` represents scaling factor `lr`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`t` represents updating step while :math:`beta_1^t` and :math:`beta_2^t` represent `beta1_power` and
    `beta2_power`, :math:`\alpha` represents `learning_rate`, :math:`w` represents `var`, :math:`\epsilon` represents
    `epsilon`.

    Args:
        use_locking (bool): Whether to enable a lock to protect updating variable tensors.
            If True, updating of the var, m, and v tensors will be protected by a lock.
            If False, the result is unpredictable. Default: False.
        use_nesterov (bool): Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients.
            If True, updates the gradients using NAG.
            If False, updates the gradients without using NAG. Default: False.

    Inputs:
        - **var** (Tensor) - Weights to be updated.
        - **m** (Tensor) - The 1st moment vector in the updating formula.
        - **v** (Tensor) - the 2nd moment vector in the updating formula.
        - **beta1_power** (float) - :math:`beta_1^t` in the updating formula.
        - **beta2_power** (float) - :math:`beta_2^t` in the updating formula.
        - **lr** (float) - :math:`l` in the updating formula.
        - **beta1** (float) - The exponential decay rate for the 1st moment estimates.
        - **beta2** (float) - The exponential decay rate for the 2nd moment estimates.
        - **epsilon** (float) - Term added to the denominator to improve numerical stability.
        - **gradient** (Tensor) - Gradients.

    Outputs:
        Tuple of 3 Tensor, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **m** (Tensor) - The same shape and data type as `m`.
        - **v** (Tensor) - The same shape and data type as `v`.
    """

    @prim_attr_register
    def __init__(self, use_locking=False, use_nesterov=False):
        validator.check_type("use_locking", use_locking, [bool])
        validator.check_type("use_nesterov", use_nesterov, [bool])

    def infer_shape(self, var_shape, m_shape, v_shape, beta1_power_shape, beta2_power_shape, lr_shape,
                    beta1_shape, beta2_shape, epsilon_shape, grad_shape):
        validator.check_param_equal("var_shape", var_shape, "m_shape", m_shape)
        validator.check_param_equal("var_shape", var_shape, "v_shape", v_shape)
        validator.check_param_equal("var_shape", var_shape, "grad_shape", grad_shape)
        return var_shape, m_shape, v_shape

    def infer_dtype(self, var_dtype, m_dtype, v_dtype, beta1_power_dtype, beta2_power_dtype, lr_dtype,
                    beta1_dtype, beta2_dtype, epsilon_dtype, grad_dtype):
        args = {"var_dtype": var_dtype, "m_dtype": m_dtype, "v_dtype": v_dtype, "grad_dtype": grad_dtype}
        validator.check_type_same(args, mstype.number_type)

        args = {"beta1_power_dtype": beta1_power_dtype, "beta2_power_dtype": beta2_power_dtype, 'lr_dtype': lr_dtype,
                "beta1_dtype": beta1_dtype, "beta2_dtype": beta2_dtype, "epsilon_dtype": epsilon_dtype}
        validator.check_type_same(args, [mstype.float16, mstype.float32])
        return var_dtype, m_dtype, v_dtype


class BinaryCrossEntropy(PrimitiveWithInfer):
    r"""
    Computes the Binary Cross Entropy between the target and the output.

    Note:
        Sets input as :math:`x`, input label as :math:`y`, output as :math:`\ell(x, y)`.
        Let,

        .. math::
            L = \{l_1,\dots,l_N\}^\top, \quad
            l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

        Then,

        .. math::
            \ell(x, y) = \begin{cases}
            L, & \text{if reduction} = \text{'none';}\\
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
            \end{cases}

    Args:
        reduction (str): Specifies the reduction to apply to the output.
            Its value should be one of 'none', 'mean', 'sum'. Default: 'mean'.

    Inputs:
        - **input_x** (Tensor) - The input Tensor.
        - **input_y** (Tensor) - The label Tensor which has same shape as `input_x`.
        - **weight** (Tensor, optional) - A rescaling weight applied to the loss of each batch element.
          And it should have same shape as `input_x`. Default: None.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and same shape as `input_x`.
        Otherwise it is a scalar.
    """

    @prim_attr_register
    def __init__(self, reduction='mean'):
        self.reduction = validator.check_string('reduction', reduction, ['none', 'mean', 'sum'])

    def infer_shape(self, x_shape, y_shape, weight_shape):
        validator.check_param_equal('x_shape', x_shape, 'y_shape', y_shape)
        if weight_shape:
            validator.check_param_equal('y_shape', y_shape, 'weight_shape', weight_shape)
        if self.reduction in ('mean', 'sum'):
            shape = []
        else:
            shape = x_shape
        return shape

    def infer_dtype(self, x_type, y_type, weight_type):
        args = {'x_type': x_type, 'y_type': y_type}
        validator.check_type_same(args, (mstype.float16, mstype.float32))
        if weight_type:
            validator.check_two_types_same('x_type', x_type, 'weight_type', weight_type)
        return x_type


class SparseApplyAdagrad(PrimitiveWithInfer):
    r"""
    Update relevant entries according to the adagrad scheme.

    .. math::
            accum += grad * grad
    .. math::
            var -= lr * grad * (1 / sqrt(accum))

    Args:
        lr (float): Learning rate.
        use_locking (bool): If True, updating of the var and accum tensors will be protected. Default: False.

    Inputs:
        - **var** (Tensor) - Variable to be updated. The type must be float32.
        - **accum** (Tensor) - Accum to be updated. The shape must be the same as `var`'s shape,
          the type must be float32.
        - **grad** (Tensor) - Gradient. The shape must be the same as `var`'s shape
          except first dimension, the type must be float32.
        - **indices** (Tensor) - A vector of indices into the first dimension of `var` and `accum`.
          The shape of `indices` must be the same as `grad` in first dimension, the type must be int32.

    Outputs:
        Tensor, has the same shape and type as `var`.
    """

    @prim_attr_register
    def __init__(self, lr, use_locking=False):
        self.lr = validator.check_type("lr", lr, [float])
        self.use_locking = validator.check_type("use_locking", use_locking, [bool])

    def infer_shape(self, var_shape, accum_shape, grad_shape, indices_shape):
        validator.check_param_equal('var shape', var_shape, 'accum shape', accum_shape)
        validator.check_param_equal('len of var shape', len(var_shape), 'len of grad shape', len(grad_shape))
        if len(var_shape) > 1:
            validator.check_param_equal('var_shape', var_shape[1:], 'grad_shape', grad_shape[1:])
        validator.check_integer("len of indices shape", len(indices_shape), 1, Rel.EQ)
        validator.check('the first dimension of grad', grad_shape[0],
                        'the shape of indices', indices_shape[0], Rel.EQ)
        return var_shape

    def infer_dtype(self, var_type, accum_type, grad_type, indices_type):
        validator.check_subclass("var_type", var_type, mstype.tensor)
        validator.check_subclass("accum_type", accum_type, mstype.tensor)
        validator.check_subclass("grad_type", grad_type, mstype.tensor)
        validator.check_subclass("indices_type", indices_type, mstype.tensor)
        args = {'var_type': var_type, 'accum_type': accum_type, 'grad_type': grad_type}
        validator.check_type_same(args, (mstype.float32,))
        validator.check_typename('indices_type', indices_type, [mstype.int32])
        return var_type


class LARSUpdate(PrimitiveWithInfer):
    """
    Conduct lars (layer-wise adaptive rate scaling) update on the square sum of gradient.

    Args:
        epsilon (float): Term added to the denominator to improve numerical stability. Default: 1e-05.
        hyperpara (float): Trust coefficient for calculating the local learning rate. Default: 0.001.
        use_clip (bool): Whether to use clip operation for calculating the local learning rate. Default: False.

    Inputs:
        - **weight** (Tensor) - The weight to be updated.
        - **gradient** (Tensor) - The gradient of weight, which has the same shape and dtype with weight.
        - **norm_weight** (Tensor) - A scalar tensor, representing the square sum of weight.
        - **norm_gradient** (Tensor) - A scalar tensor, representing the square sum of gradient.
        - **weight_decay** (Union[Number, Tensor]) - Weight decay. It should be a scalar tensor or number.
        - **learning_rate** (Union[Number, Tensor]) - Learning rate. It should be a scalar tensor or number.

    Outputs:
        Tensor, representing the new gradient.
    """

    @prim_attr_register
    def __init__(self, epsilon=1e-05, hyperpara=0.001, use_clip=False):
        """init"""
        validator.check_type("epsilon", epsilon, [float])
        validator.check_type("hyperpara", hyperpara, [float])
        validator.check_type("use_clip", use_clip, [bool])

    def infer_shape(self, weight_shape, gradient_shape, norm_weight_shape, norm_gradient_shape, weight_decay_shape,
                    learning_rate_shape):
        validator.check_param_equal("Weight shape", weight_shape, "gradient shape", gradient_shape)
        validator.check_param_equal("Norm weight shape", norm_weight_shape, "norm gradient shape", norm_gradient_shape)
        shp_len = len(weight_decay_shape)
        validator.check_shape_length("Weight decay's shape", shp_len, 1, Rel.LE)
        if shp_len == 1:
            validator.check_integer("Weight decay's shape", weight_decay_shape[0], 1, Rel.EQ)
        shp_len = len(learning_rate_shape)
        validator.check_shape_length("Learning rate's shape", shp_len, 1, Rel.LE)
        if shp_len == 1:
            validator.check_integer("Learning rate's shape", learning_rate_shape[0], 1, Rel.EQ)
        return weight_shape

    def infer_dtype(self, weight_dtype, gradient_dtype, norm_weight_dtype, norm_gradient_dtype,
                    weight_decay_dtype, learning_rate_dtype):
        args = {"Weight dtype": weight_dtype, "gradient dtype": gradient_dtype, "norm weight dtype": norm_weight_dtype,
                "norm gradient dtype": norm_gradient_dtype}
        validator.check_type_same(args, [mstype.float16, mstype.float32, mstype.int16, mstype.int32])
        validator.check_args_tensor(args)
        validator.check_typename("weight_decay_dtype", weight_decay_dtype,
                                 [mstype.float16, mstype.float32, mstype.float64])
        validator.check_typename("learning_rate_dtype", learning_rate_dtype,
                                 [mstype.float16, mstype.float32, mstype.float64])
        return weight_dtype


class ApplyFtrl(PrimitiveWithInfer):
    """
    Update relevant entries according to the FTRL scheme.

    Args:
        use_locking (bool): Use locks for update operation if True . Default: False.

    Inputs:
        - **var** (Tensor): The variable to be updated.
        - **accum** (Tensor): The accum to be updated, must be same type and shape as `var`.
        - **linear** (Tensor): The linear to be updated, must be same type and shape as `var`.
        - **grad** (Tensor): Gradient.
        - **lr** (Union[Number, Tensor]): The learning rate value, must be positive. Default: 0.001.
        - **l1** (Union[Number, Tensor]): l1 regularization strength, must be greater than or equal to zero.
          Default: 0.0.
        - **l2** (Union[Number, Tensor]): l2 regularization strength, must be greater than or equal to zero.
          Default: 0.0.
        - **lr_power** (Union[Number, Tensor]): Learning rate power controls how the learning rate decreases
          during training, must be less than or equal to zero. Use fixed learning rate if lr_power is zero.
          Default: -0.5.

    Outputs:
        Tensor, representing the updated var.
    """

    @prim_attr_register
    def __init__(self, use_locking=False):
        self.init_prim_io_names(inputs=['var', 'accum', 'linear', 'grad', 'lr', 'l1', 'l2', 'lr_power'],
                                outputs=['output'])
        self.use_locking = validator.check_type("use_locking", use_locking, [bool])

    def infer_shape(self, var_shape, accum_shape, linear_shape, grad_shape, lr_shape, l1_shape, l2_shape,
                    lr_power_shape):
        validator.check_param_equal('var shape', var_shape, 'accum shape', accum_shape)
        validator.check_param_equal('var shape', var_shape, 'linear shape', linear_shape)
        return var_shape

    def infer_dtype(self, var_type, accum_type, linear_type, grad_type, lr_type, l1_type, l2_type, lr_power_type):
        validator.check_subclass("var_type", var_type, mstype.tensor)
        validator.check_subclass("accum_type", accum_type, mstype.tensor)
        validator.check_subclass("linear_type", linear_type, mstype.tensor)
        validator.check_subclass("grad_type", grad_type, mstype.tensor)
        args = {'var_type': var_type, 'accum_type': accum_type, 'linear_type': linear_type, 'grad_type': grad_type}
        validator.check_type_same(args, (mstype.float32, mstype.float16))

        validator.check_typename("lr", lr_type, [mstype.float16, mstype.float32])
        validator.check_typename("l1", l1_type, [mstype.float16, mstype.float32])
        validator.check_typename("l2", l2_type, [mstype.float16, mstype.float32])
        validator.check_typename("lr_power", lr_power_type, [mstype.float16, mstype.float32])
        return var_type


class ExtractImagePatches(PrimitiveWithInfer):
    """
    Extract patches from images.
    The input tensor must be a 4-D tensor and the data format is NHWC.

    Args:
        ksizes (Union[tuple[int], list[int]]): The size of sliding window, should be a tuple or list of int,
            and the format is [1, ksize_row, ksize_col, 1].
        strides (Union[tuple[int], list[int]]): Distance between the centers of the two consecutive patches,
            should be a tuple or list of int, and the format is [1, stride_row, stride_col, 1].
        rates (Union[tuple[int], list[int]]): In each extracted patch, the gap between the corresponding dim
            pixel positions, should be a tuple or list of int, and the format is [1, rate_row, rate_col, 1].
        padding (str): The type of padding algorithm, is a string whose value is "same" or "valid",
            not case sensitive. Default: "valid".

            - same: Means that the patch can take the part beyond the original image, and this part is filled with 0.

            - valid: Means that the patch area taken must be completely contained in the original image.

    Inputs:
        - **input_x** (Tensor) - A 4-D tensor whose shape is [in_batch, in_row, in_col, in_depth] and
          data type is int8, float16, uint8.

    Outputs:
        Tensor, a 4-D tensor whose data type is same as 'input_x',
        and the shape is [out_batch, out_row, out_col, out_depth], the out_batch is same as the in_batch.
    """

    @prim_attr_register
    def __init__(self, ksizes, strides, rates, padding="valid"):
        """init"""
        validator.check_type("ksizes", ksizes, [tuple, list])
        validator.check_type("strides", strides, [tuple, list])
        validator.check_type("rates", rates, [tuple, list])
        self.padding = validator.check_string('padding', padding.upper(), ['VALID', 'SAME'])
        self.add_prim_attr("padding", self.padding)

        if len(ksizes) != 4 or ksizes[0] != 1 or ksizes[3] != 1:
            raise ValueError("The format of ksizes should be [1, ksize_row, ksize_col, 1], "
                             f"but got {ksizes}.")
        if not isinstance(ksizes[1], int) or not isinstance(ksizes[2], int) or \
                ksizes[1] < 1 or ksizes[2] < 1:
            raise ValueError("The ksize_row and ksize_col in ksizes should be an positive integer number, "
                             f"but got ksize_row is {ksizes[1]}, ksize_col is {ksizes[2]}")

        if len(strides) != 4 or strides[0] != 1 or strides[3] != 1:
            raise ValueError("The format of strides should be [1, stride_row, stride_col, 1], "
                             f"but got {strides}.")
        if not isinstance(strides[1], int) or not isinstance(strides[2], int) or \
                strides[1] < 1 or strides[2] < 1:
            raise ValueError("The stride_row and stride_col in strides should be an positive integer number, "
                             f"but got stride_row is {strides[1]}, stride_col is {strides[2]}")

        if len(rates) != 4 or rates[0] != 1 or rates[3] != 1:
            raise ValueError("The format of rates should be [1, rate_row, rate_col, 1], "
                             f"but got {rates}.")
        if not isinstance(rates[1], int) or not isinstance(rates[2], int) or \
                rates[1] < 1 or rates[2] < 1:
            raise ValueError("The rate_row and rate_col in rates should be an positive integer number, "
                             f"but got rate_row is {rates[1]}, rate_col is {rates[2]}")

    def infer_shape(self, input_x):
        in_batch, in_row, in_col, in_depth = input_x
        _, ksize_row, ksize_col, _ = self.ksizes
        _, stride_row, stride_col, _ = self.strides
        _, rate_row, rate_col, _ = self.rates
        if len(input_x) != 4:
            raise ValueError("The `input_x` should be a 4-D tensor, "
                             f"but got a {len(input_x)}-D tensor whose shape is {input_x}")

        out_batch = in_batch
        out_depth = ksize_row * ksize_col * in_depth

        if self.padding == "VALID":
            out_row = \
                (in_row - (ksize_row + (ksize_row - 1) * (rate_row - 1))) // stride_row + 1
            out_col = \
                (in_col - (ksize_col + (ksize_col - 1) * (rate_col - 1))) // stride_col + 1
        else:
            out_row = (in_row - 1) // stride_row + 1
            out_col = (in_col - 1) // stride_col + 1

        out_shape = [out_batch, out_row, out_col, out_depth]
        return out_shape

    def infer_dtype(self, input_x):
        validator.check_subclass("input_x", input_x, mstype.tensor)
        validator.check_typename("input_x_dtype", input_x, (mstype.int8, mstype.float16, mstype.float32))
        return input_x

# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""activation"""
import numpy as np

from mindspore._checkparam import Validator as validator
from mindspore._extends import cell_attr_register
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from ..cell import Cell

__all__ = ['Softmax',
           'LogSoftmax',
           'ReLU',
           'ReLU6',
           'Tanh',
           'GELU',
           'FastGelu',
           'Sigmoid',
           'PReLU',
           'get_activation',
           'LeakyReLU',
           'HSigmoid',
           'HSwish',
           'ELU',
           'LogSigmoid',
           'SoftShrink',
           'HShrink',
           ]


class Softmax(Cell):
    r"""
    Softmax activation function.

    Applies the Softmax function to an n-dimensional input Tensor.

    The input is a Tensor of logits transformed with exponential function and then
    normalized to lie in range [0, 1] and sum up to 1.

    Softmax is defined as:

    .. math::
        \text{softmax}(x_{i}) =  \frac{\exp(x_i)}{\sum_{j=0}^{n-1}\exp(x_j)},

    where :math:`x_{i}` is the :math:`i`-th slice in the given dimension of the input Tensor.

    Args:
        axis (Union[int, tuple[int]]): The axis to apply Softmax operation, -1 means the last dimension. Default: -1.

    Inputs:
        - **x** (Tensor) - The input of Softmax with data type of float16 or float32.

    Outputs:
        Tensor, which has the same type and shape as `x` with values in the range[0,1].

    Raises:
        TypeError: If `axis` is neither an int nor a tuple.
        TypeError: If dtype of `x` is neither float16 nor float32.
        ValueError: If `axis` is a tuple whose length is less than 1.
        ValueError: If `axis` is a tuple whose elements are not all in range [-len(x), len(x)).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> softmax = nn.Softmax()
        >>> output = softmax(x)
        >>> print(output)
        [0.03168 0.01166 0.0861  0.636   0.2341 ]
    """

    def __init__(self, axis=-1):
        """Initialize Softmax."""
        super(Softmax, self).__init__()
        self.softmax = P.Softmax(axis)

    def construct(self, x):
        return self.softmax(x)


class LogSoftmax(Cell):
    r"""
    LogSoftmax activation function.

    Applies the LogSoftmax function to n-dimensional input tensor.

    The input is transformed by the Softmax function and then by the log function to lie in range[-inf,0).

    Logsoftmax is defined as:

    .. math::

        \text{logsoftmax}(x_i) = \log \left(\frac{\exp(x_i)}{\sum_{j=0}^{n-1} \exp(x_j)}\right),

    where :math:`x_{i}` is the :math:`i`-th slice in the given dimension of the input Tensor.

    Args:
        axis (int): The axis to apply LogSoftmax operation, -1 means the last dimension. Default: -1.

    Inputs:
        - **x** (Tensor) - The input of LogSoftmax, with float16 or float32 data type.

    Outputs:
        Tensor, which has the same type and shape as the input as `x` with values in the range[-inf,0).

    Raises:
        TypeError: If `axis` is not an int.
        TypeError: If dtype of `x` is neither float16 nor float32.
        ValueError: If `axis` is not in range [-len(x), len(x)).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> log_softmax = nn.LogSoftmax()
        >>> output = log_softmax(x)
        >>> print(output)
        [[-5.00672150e+00 -6.72150636e-03 -1.20067215e+01]
         [-7.00091219e+00 -1.40009127e+01 -9.12250078e-04]]
    """

    def __init__(self, axis=-1):
        """Initialize LogSoftmax."""
        super(LogSoftmax, self).__init__()
        self.log_softmax = P.LogSoftmax(axis)

    def construct(self, x):
        return self.log_softmax(x)


class ELU(Cell):
    r"""
    Exponential Linear Uint activation function.

    Applies the exponential linear unit function element-wise.
    The activation function is defined as:

    .. math::
        E_{i} =
        \begin{cases}
        x, &\text{if } x \geq 0; \cr
        \text{alpha} * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    The picture about ELU looks like this `ELU <https://en.wikipedia.org/wiki/
    Activation_function#/media/File:Activation_elu.svg>`_.

    Args:
        alpha (float): The coefficient of negative factor whose type is float. Default: 1.0.

    Inputs:
        - **x** (Tensor) - The input of ELU with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means,any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If `alpha` is not a float.
        TypeError: If dtype of `x` is neither float16 nor float32.
        ValueError: If `alpha` is not equal to 1.0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float32)
        >>> elu = nn.ELU()
        >>> result = elu(x)
        >>> print(result)
        [-0.63212055  -0.86466473  0.  2.  1.]
    """

    def __init__(self, alpha=1.0):
        """Initialize ELU."""
        super(ELU, self).__init__()
        self.elu = P.Elu(alpha)

    def construct(self, x):
        return self.elu(x)


class ReLU(Cell):
    r"""
    Rectified Linear Unit activation function.

    Applies the rectified linear unit function element-wise.

    .. math::

        \text{ReLU}(x) = (x)^+ = \max(0, x),

    It returns element-wise :math:`\max(0, x)`, specially, the neurons with the negative output
    will be suppressed and the active neurons will stay the same.

    The picture about ReLU looks like this `ReLU <https://en.wikipedia.org/wiki/
    Activation_function#/media/File:Activation_rectified_linear.svg>`_.

    Inputs:
        - **x** (Tensor) - The input of ReLU. The data type is Number.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is not a number.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
        >>> relu = nn.ReLU()
        >>> output = relu(x)
        >>> print(output)
        [0. 2. 0. 2. 0.]
    """

    def __init__(self):
        """Initialize ReLU."""
        super(ReLU, self).__init__()
        self.relu = P.ReLU()

    def construct(self, x):
        return self.relu(x)


class ReLU6(Cell):
    r"""
    Compute ReLU6 activation function.

    ReLU6 is similar to ReLU with a upper limit of 6, which if the inputs are greater than 6, the outputs
    will be suppressed to 6.
    It computes element-wise as

    .. math::

        \min(\max(0, x), 6).

    The input is a Tensor of any valid shape.

    Inputs:
        - **x** (Tensor) - The input of ReLU6 with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, which has the same type as `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> relu6 = nn.ReLU6()
        >>> output = relu6(x)
        >>> print(output)
        [0. 0. 0. 2. 1.]
    """

    def __init__(self):
        """Initialize ReLU6."""
        super(ReLU6, self).__init__()
        self.relu6 = P.ReLU6()

    def construct(self, x):
        return self.relu6(x)


class LeakyReLU(Cell):
    r"""
    Leaky ReLU activation function.

    LeakyReLU is similar to ReLU, but LeakyReLU has a slope that makes it not equal to 0 at x < 0.
    The activation function is defined as:

    .. math::
            \text{leaky_relu}(x) = \begin{cases}x, &\text{if } x \geq 0; \cr
            \text{alpha} * x, &\text{otherwise.}\end{cases}

    See https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf

    Args:
        alpha (Union[int, float]): Slope of the activation function at x < 0. Default: 0.2.

    Inputs:
        - **x** (Tensor) - The input of LeakyReLU.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same type and shape as the `x`.

    Raises:
        TypeError: If `alpha` is not a float or an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> leaky_relu = nn.LeakyReLU()
        >>> output = leaky_relu(x)
        >>> print(output)
        [[-0.2  4.  -1.6]
         [ 2.  -1.   9. ]]
    """

    def __init__(self, alpha=0.2):
        """Initialize LeakyReLU."""
        super(LeakyReLU, self).__init__()
        validator.check_value_type('alpha', alpha, [float, int], self.cls_name)
        self.greater_equal = P.GreaterEqual()
        self.mul = P.Mul()
        self.alpha = alpha
        self.select_op = P.Maximum()
        if self.alpha > 1:
            self.select_op = P.Minimum()

    def construct(self, x):
        alpha_array = P.Cast()(F.scalar_to_array(self.alpha), P.DType()(x))
        out = self.select_op(alpha_array * x, x)
        return out


class Tanh(Cell):
    r"""
    Tanh activation function.

    Applies the Tanh function element-wise, returns a new tensor with the hyperbolic tangent of the elements of input,
    The input is a Tensor with any valid shape.

    Tanh function is defined as:

    .. math::
        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    where :math:`x_i` is an element of the input Tensor.

    Inputs:
        - **x** (Tensor) - The input of Tanh with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3, 2, 1]), mindspore.float16)
        >>> tanh = nn.Tanh()
        >>> output = tanh(x)
        >>> print(output)
        [0.7617 0.964  0.995  0.964  0.7617]
    """

    def __init__(self):
        """Initialize Tanh."""
        super(Tanh, self).__init__()
        self.tanh = P.Tanh()

    def construct(self, x):
        return self.tanh(x)


class GELU(Cell):
    r"""
    Gaussian error linear unit activation function.

    Applies GELU function to each element of the input. The input is a Tensor with any valid shape.

    GELU is defined as:

    .. math::

        GELU(x_i) = x_i*P(X < x_i),

    where :math:`P` is the cumulative distribution function
    of standard Gaussian distribution and :math:`x_i` is the element of the input.

    The picture about GELU looks like this `GELU <https://en.wikipedia.org/wiki/
    Activation_function#/media/File:Activation_gelu.png>`_.

    Inputs:
        - **x** (Tensor) - The input of GELU with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> gelu = nn.GELU()
        >>> output = gelu(x)
        >>> print(output)
        [[-1.5880802e-01  3.9999299e+00 -3.1077917e-21]
         [ 1.9545976e+00 -2.2918017e-07  9.0000000e+00]]
    """

    def __init__(self):
        """Initialize GELU."""
        super(GELU, self).__init__()
        self.gelu = P.GeLU()

    def construct(self, x):
        return self.gelu(x)


class FastGelu(Cell):
    r"""
    Fast Gaussian error linear unit activation function.

    Applies FastGelu function to each element of the input. The input is a Tensor with any valid shape.

    FastGelu is defined as:

    .. math::
        FastGelu(x_i) = \frac {x_i} {1 + \exp(-1.702 * \left| x_i \right|)} *
                           \exp(0.851 * (x_i - \left| x_i \right|))

    where :math:`x_i` is the element of the input.

    Inputs:
        - **x** (Tensor) - The input of FastGelu with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> fast_gelu = nn.FastGelu()
        >>> output = fast_gelu(x)
        >>> print(output)
        [[-1.5418735e-01  3.9921875e+00 -9.7473649e-06]
         [ 1.9375000e+00 -1.0052517e-03  8.9824219e+00]]
    """

    def __init__(self):
        """Initialize FastGelu."""
        super(FastGelu, self).__init__()
        self.fast_gelu = P.FastGeLU()

    def construct(self, x):
        return self.fast_gelu(x)


class Sigmoid(Cell):
    r"""
    Sigmoid activation function.

    Applies sigmoid-type activation element-wise.

    Sigmoid function is defined as:

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    where :math:`x_i` is the element of the input.

    The picture about Sigmoid looks like this `Sigmoid <https://en.wikipedia.org/wiki/
    Sigmoid_function#/media/File:Logistic-curve.svg>`_.

    Inputs:
        - **x** (Tensor) - The input of Sigmoid with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> sigmoid = nn.Sigmoid()
        >>> output = sigmoid(x)
        >>> print(output)
        [0.2688  0.11914 0.5     0.881   0.7305 ]
    """

    def __init__(self):
        """Initialize Sigmoid."""
        super(Sigmoid, self).__init__()
        self.sigmoid = P.Sigmoid()

    def construct(self, x):
        return self.sigmoid(x)


class PReLU(Cell):
    r"""
    PReLU activation function.

    Applies the PReLU function element-wise.

    PReLU is defined as:

    .. math::

        prelu(x_i)= \max(0, x_i) + w * \min(0, x_i),

    where :math:`x_i` is an element of an channel of the input.

    Here :math:`w` is a learnable parameter with a default initial value 0.25.
    Parameter :math:`w` has dimensionality of the argument channel. If called without argument
    channel, a single parameter :math:`w` will be shared across all channels.

    The picture about PReLU looks like this `PReLU <https://en.wikipedia.org/wiki/
    Activation_function#/media/File:Activation_prelu.svg>`_.

    Args:
        channel (int): The elements number of parameter.
          It could be an int, and the value is 1 or the channels number of input tensor `x`. Default: 1.
        w (Union[float, list, Tensor]): The initial value of parameter. It could be a float, a float list or
          a tensor has the same dtype as the input tensor `x`. Default: 0.25.

    Inputs:
        - **x** (Tensor) - The input of PReLU with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same dtype and shape as the `x`.

    Raises:
        TypeError: If `channel` is not an int.
        TypeError: If `w` is not one of a float, a float list, a float Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.
        ValueError: If the `x` is a 0-D or 1-D Tensor on Ascend.
        ValueError: If `channel` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([[[[0.1, 0.6], [0.9, 0.9]]]]), mindspore.float32)
        >>> prelu = nn.PReLU()
        >>> output = prelu(x)
        >>> print(output)
        [[[[0.1 0.6]
           [0.9 0.9]]]]

    """
    @cell_attr_register(attrs="")
    def __init__(self, channel=1, w=0.25):
        """Initialize PReLU."""
        super(PReLU, self).__init__()
        validator.check_positive_int(channel, 'channel', self.cls_name)
        if isinstance(w, (float, np.float32)):
            tmp = np.empty((channel,), dtype=np.float32)
            tmp.fill(w)
            w = Tensor(tmp, dtype=mstype.float32)
        elif isinstance(w, list):
            if len(w) != channel:
                raise ValueError(f"For '{self.cls_name}', the length of 'w' should be equal to the 'channel' when "
                                 f"the 'w' is a list, but got the length of 'w': {len(w)}, the 'channel': {channel}.")

            for i in w:
                if not isinstance(i, (float, np.float32)):
                    raise ValueError(f"For '{self.cls_name}', all elements in 'w' should be "
                                     f"float when the 'w' is a list, but got {i}.")
            w = Tensor(w, dtype=mstype.float32)
        elif isinstance(w, Tensor):
            if w.dtype not in (mstype.float16, mstype.float32):
                raise ValueError(f"For '{self.cls_name}', the dtype of 'w' should be float16 or "
                                 f"float32 when the 'w' is a tensor, but got {w.dtype}.")
            if len(w.shape) != 1 or w.shape[0] != channel:
                raise ValueError(f"For '{self.cls_name}', the dimension of 'w' should be 1, and the elements number "
                                 f"should be equal to the 'channel' when the 'w' is a tensor, "
                                 f"but got 'w' shape {w.shape}, the 'channel' {channel}.")
        else:
            raise TypeError(f"For '{self.cls_name}', the 'w' only supported float, list and tensor, "
                            f"but got {type(w).__name__}.")
        self.w = Parameter(w, name='a')
        self.prelu = P.PReLU()
        self.relu = P.ReLU()
        self.assign = P.Assign()

    def construct(self, x):
        u = self.relu(self.w)
        v = self.prelu(x, F.cast(u, x.dtype))
        if self.training:
            self.assign(self.w, u)
        return v


class HSwish(Cell):
    r"""
    Hard swish activation function.

    Applies hswish-type activation element-wise. The input is a Tensor with any valid shape.

    Hard swish is defined as:

    .. math::
        \text{hswish}(x_{i}) = x_{i} * \frac{ReLU6(x_{i} + 3)}{6},

    where :math:`x_{i}` is the :math:`i`-th slice in the given dimension of the input Tensor.

    Inputs:
        - **x** (Tensor) - The input of HSwish, data type must be float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> hswish = nn.HSwish()
        >>> result = hswish(x)
        >>> print(result)
        [-0.3333  -0.3333  0  1.666  0.6665]
    """

    def __init__(self):
        """Initialize HSwish."""
        super(HSwish, self).__init__()
        self.hswish = P.HSwish()

    def construct(self, x):
        return self.hswish(x)


class HSigmoid(Cell):
    r"""
    Hard sigmoid activation function.

    Applies hard sigmoid activation element-wise. The input is a Tensor with any valid shape.

    Hard sigmoid is defined as:

    .. math::
        \text{hsigmoid}(x_{i}) = max(0, min(1, \frac{x_{i} + 3}{6})),

    where :math:`x_{i}` is the :math:`i`-th slice in the given dimension of the input Tensor.

    Inputs:
        - **input_x** (Tensor) - The input of HSigmoid. The shape is :math:`(N,*)` where :math:`*` means, any number of
          additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> hsigmoid = nn.HSigmoid()
        >>> result = hsigmoid(x)
        >>> print(result)
        [0.3333 0.1666 0.5    0.8335 0.6665]
    """

    def __init__(self):
        """Initialize HSigmoid."""
        super(HSigmoid, self).__init__()
        self.hsigmoid = P.HSigmoid()

    def construct(self, input_x):
        return self.hsigmoid(input_x)


class LogSigmoid(Cell):
    r"""
    Logsigmoid activation function.

    Applies logsigmoid activation element-wise. The input is a Tensor with any valid shape.

    Logsigmoid is defined as:

    .. math::
        \text{logsigmoid}(x_{i}) = log(\frac{1}{1 + \exp(-x_i)}),

    where :math:`x_{i}` is the element of the input.

    Inputs:
        - **x** (Tensor) - The input of LogSigmoid with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.LogSigmoid()
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> output = net(x)
        >>> print(output)
        [-0.31326166 -0.12692806 -0.04858734]
    """

    def __init__(self):
        """Initialize LogSigmoid."""
        super(LogSigmoid, self).__init__()
        self.mul = P.Mul()
        self.exp = P.Exp()
        self.add = P.Add()
        self.rec = P.Reciprocal()
        self.log = P.Log()

    def construct(self, input_x):
        neg_input = self.mul(input_x, -1)
        exp_neg_input = self.exp(neg_input)
        exp_neg_input_1 = self.add(exp_neg_input, 1)
        rec_exp_neg_input_1 = self.rec(exp_neg_input_1)
        ret = self.log(rec_exp_neg_input_1)
        return ret

class SoftShrink(Cell):
    r"""
    Applies the soft shrinkage function elementwise.

    .. math::
        \text{SoftShrink}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` must be no less than zero value for the Softshrink formulation. Default: 0.5.

    Inputs:
        - **input_x** (Tensor) - The input of SoftShrink with data type of float16 or float32.
          Any number of additional dimensions.

    Outputs:
        Tensor, has the same shape and data type as `input_x`.

    Raises:
        TypeError: If lambd is not a float.
        TypeError: If input_x is not a Tensor.
        TypeError: If dtype of input_x is neither float16 nor float32.
        ValueError: If lambd is less than 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([[ 0.5297,  0.7871,  1.1754], [ 0.7836,  0.6218, -1.1542]]), mstype.float16)
        >>> softshrink = nn.SoftShrink()
        >>> output = softshrink(input_x)
        >>> print(output)
        [[ 0.02979  0.287    0.676  ]
         [ 0.2837   0.1216  -0.6543 ]]
    """

    def __init__(self, lambd=0.5):
        super(SoftShrink, self).__init__()
        self.softshrink = P.SoftShrink(lambd)

    def construct(self, input_x):
        output = self.softshrink(input_x)
        return output

class HShrink(Cell):
    r"""
    Applies the hard shrinkage function element-wise, each element complies the follow function:

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd (float): The value for the HardShrink formulation. Default: 0.5

    Inputs:
        - **input_x** (Tensor) - The input of HardShrink with data type of float16 or float32.

    Outputs:
        Tensor, the same shape and data type as the input.

    Supported Platforms:
        ``Ascend``

    Raises:
        TypeError: If `lambd` is not a float.
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Examples:
        >>> input_x = Tensor(np.array([[ 0.5,  1,  2.0],[0.0533,0.0776,-2.1233]]),mstype.float32)
        >>> hshrink = nn.HShrink()
        >>> output = hshrink(input_x)
        >>> print(output)
        [[ 0.      1.      2.    ]
        [ 0.      0.     -2.1233]]
    """

    def __init__(self, lambd=0.5):
        super(HShrink, self).__init__()
        self.hshrink = P.HShrink(lambd)

    def construct(self, input_x):
        return self.hshrink(input_x)


_activation = {
    'softmax': Softmax,
    'logsoftmax': LogSoftmax,
    'relu': ReLU,
    'relu6': ReLU6,
    'tanh': Tanh,
    'gelu': GELU,
    'fast_gelu': FastGelu,
    'elu': ELU,
    'sigmoid': Sigmoid,
    'prelu': PReLU,
    'leakyrelu': LeakyReLU,
    'hswish': HSwish,
    'hsigmoid': HSigmoid,
    'logsigmoid': LogSigmoid,
    'softshrink': SoftShrink,
    'hshrink': HShrink,
}


def get_activation(name, prim_name=None):
    """
    Gets the activation function.

    Args:
        name (str): The name of the activation function.

    Returns:
        Function, the activation function.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> sigmoid = nn.get_activation('sigmoid')
        >>> print(sigmoid)
        Sigmoid<>
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if name is None:
        return None

    if name not in _activation:
        raise KeyError(f"{msg_prefix} 'name' should be in {_activation}, but got {name}.")
    return _activation[name]()

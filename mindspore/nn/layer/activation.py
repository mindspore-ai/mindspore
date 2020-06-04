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
"""activation"""
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore._extends import cell_attr_register
from mindspore.ops import resolved_ops as RO
from ..cell import Cell


__all__ = ['Softmax',
           'LogSoftmax',
           'ReLU',
           'ReLU6',
           'Tanh',
           'GELU',
           'Sigmoid',
           'PReLU',
           'get_activation',
           'LeakyReLU',
           'HSigmoid',
           'HSwish',
           'ELU',
           'LogSigmoid',
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

    where :math:`x_{i}` is the :math:`i`-th slice along the given dim of the input Tensor.

    Args:
        axis (Union[int, tuple[int]]): The axis to apply Softmax operation, -1 means the last dimension. Default: -1.

    Inputs:
        - **x** (Tensor) - The input of Softmax.

    Outputs:
        Tensor, which has the same type and shape as `x` with values in the range[0,1].

    Examples:
        >>> input_x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> softmax = nn.Softmax()
        >>> softmax(input_x)
        [0.03168  0.01166  0.0861  0.636  0.2341]
    """

    def __init__(self, axis=-1):
        super(Softmax, self).__init__()
        self.softmax = RO.Softmax(axis)

    def construct(self, x):
        return self.softmax(x)


class LogSoftmax(Cell):
    r"""
    LogSoftmax activation function.

    Applies the LogSoftmax function to n-dimensional input tensor.

    The input is transformed with Softmax function and then with log function to lie in range[-inf,0).

    Logsoftmax is defined as:
    :math:`\text{logsoftmax}(x_i) = \log \left(\frac{\exp(x_i)}{\sum_{j=0}^{n-1} \exp(x_j)}\right)`,
    where :math:`x_{i}` is the :math:`i`-th slice along the given dim of the input Tensor.

    Args:
        axis (int): The axis to apply LogSoftmax operation, -1 means the last dimension. Default: -1.

    Inputs:
        - **x** (Tensor) - The input of LogSoftmax.

    Outputs:
        Tensor, which has the same type and shape as the input as `x` with values in the range[-inf,0).

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> log_softmax = nn.LogSoftmax()
        >>> log_softmax(input_x)
        [[-5.00672150e+00 -6.72150636e-03 -1.20067215e+01]
         [-7.00091219e+00 -1.40009127e+01 -9.12250078e-04]]
    """

    def __init__(self, axis=-1):
        super(LogSoftmax, self).__init__()
        self.log_softmax = RO.LogSoftmax(axis)

    def construct(self, x):
        return self.log_softmax(x)


class ELU(Cell):
    r"""
    Exponential Linear Uint activation function.

    Applies the exponential linear unit function element-wise.
    The activation function defined as:

    .. math::
        E_{i} =
        \begin{cases}
        x, &\text{if } x \geq 0; \cr
        \text{alpha} * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    Args:
        alpha (float): The coefficient of negative factor whose type is float. Default: 1.0.

    Inputs:
        - **input_data** (Tensor) - The input of ELU.

    Outputs:
        Tensor, with the same type and shape as the `input_data`.

    Examples:
        >>> input_x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float32)
        >>> elu = nn.ELU()
        >>> elu(input_x)

    """

    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.elu = P.Elu(alpha)

    def construct(self, x):
        return self.elu(x)


class ReLU(Cell):
    r"""
    Rectified Linear Unit activation function.

    Applies the rectified linear unit function element-wise. It returns
    element-wise :math:`\max(0, x)`, specially, the neurons with the negative output
    will suppressed and the active neurons will stay the same.

    Inputs:
        - **input_data** (Tensor) - The input of ReLU.

    Outputs:
        Tensor, with the same type and shape as the `input_data`.

    Examples:
        >>> input_x = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
        >>> relu = nn.ReLU()
        >>> relu(input_x)
        [0.  2.  0.  2.  0.]
    """

    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = P.ReLU()

    def construct(self, x):
        return self.relu(x)


class ReLU6(Cell):
    r"""
    Compute ReLU6 activation function.

    ReLU6 is similar to ReLU with a upper limit of 6, which if the inputs are greater than 6, the outputs
    will be suppressed to 6.
    It computes element-wise as :math:`\min(\max(0, x), 6)`. The input is a Tensor of any valid shape.

    Inputs:
        - **input_data** (Tensor) - The input of ReLU6.

    Outputs:
        Tensor, which has the same type with `input_data`.

    Examples:
        >>> input_x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> relu6 = nn.ReLU6()
        >>> relu6(input_x)
        [0.  0.  0.  2.  1.]
    """

    def __init__(self):
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
        alpha (float): Slope of the activation function at x < 0. Default: 0.2.

    Inputs:
        - **input_x** (Tensor) - The input of LeakyReLU.

    Outputs:
        Tensor, has the same type and shape with the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> leaky_relu = nn.LeakyReLU()
        >>> leaky_relu(input_x)
        [[-0.2  4.  -1.6]
         [ 2   -1.   9.]]
    """

    def __init__(self, alpha=0.2):
        super(LeakyReLU, self).__init__()
        self.greater_equal = P.GreaterEqual()
        self.mul = P.Mul()
        self.alpha = alpha

    def construct(self, x):
        alpha = P.Cast()(F.scalar_to_array(self.alpha), P.DType()(x))
        if self.alpha <= 1:
            out = P.Maximum()(alpha * x, x)
        else:
            out = P.Minimum()(alpha * x, x)
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
        - **input_data** (Tensor) - The input of Tanh.

    Outputs:
        Tensor, with the same type and shape as the `input_data`.

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3, 2, 1]), mindspore.float16)
        >>> tanh = nn.Tanh()
        >>> tanh(input_x)
        [0.7617  0.964  0.995  0.964 0.7617]
    """

    def __init__(self):
        super(Tanh, self).__init__()
        self.tanh = RO.Tanh()

    def construct(self, x):
        return self.tanh(x)


class GELU(Cell):
    r"""
    Gaussian error linear unit activation function.

    Applies GELU function to each element of the input. The input is a Tensor with any valid shape.

    GELU is defined as:
    :math:`GELU(x_i) = x_i*P(X < x_i)`, where :math:`P` is the cumulative distribution function
    of standard Gaussian distribution and :math:`x_i` is the element of the input.

    Inputs:
        - **input_data** (Tensor) - The input of Tanh.

    Outputs:
        Tensor, with the same type and shape as the `input_data`.

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> gelu = nn.GELU()
        >>> gelu(input_x)
        [[-1.5880802e-01  3.9999299e+00 -3.1077917e-21]
         [ 1.9545976e+00 -2.2918017e-07  9.0000000e+00]]
    """

    def __init__(self):
        super(GELU, self).__init__()
        self.gelu = RO.Gelu()

    def construct(self, x):
        return self.gelu(x)


class Sigmoid(Cell):
    r"""
    Sigmoid activation function.

    Applies sigmoid-type activation element-wise.

    Sigmoid function is defined as:
    :math:`\text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)}`, where :math:`x_i` is the element of the input.

    Inputs:
        - **input_data** (Tensor) - The input of Tanh.

    Outputs:
        Tensor, with the same type and shape as the `input_data`.

    Examples:
        >>> input_x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> sigmoid = nn.Sigmoid()
        >>> sigmoid(input_x)
        [0.2688  0.11914  0.5  0.881  0.7305]
    """

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sigmoid = P.Sigmoid()

    def construct(self, x):
        return self.sigmoid(x)


class PReLU(Cell):
    r"""
    PReLU activation function.

    Applies the PReLU function element-wise.

    PReLU is defined as: :math:`prelu(x_i)= \max(0, x_i) + w * \min(0, x_i)`, where :math:`x_i`
    is an element of an channel of the input.

    Here :math:`w` is an learnable parameter with default initial value 0.25.
    Parameter :math:`w` has dimensionality of the argument channel. If called without argument
    channel, a single parameter :math:`w` will be shared across all channels.

    Args:
        channel (int): The dimension of input. Default: 1.
        w (float): The initial value of w. Default: 0.25.

    Inputs:
        - **input_data** (Tensor) - The input of PReLU.

    Outputs:
        Tensor, with the same type and shape as the `input_data`.

    Examples:
        >>> input_x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float32)
        >>> prelu = nn.PReLU()
        >>> prelu(input_x)

    """
    @cell_attr_register(attrs="")
    def __init__(self, channel=1, w=0.25):
        super(PReLU, self).__init__()
        if isinstance(w, (np.float32, float)):
            tmp = np.empty((channel,), dtype=np.float32)
            tmp.fill(w)
            w = Tensor(tmp)
        elif isinstance(w, list):
            w = Tensor(w)

        if not isinstance(w, Tensor):
            raise TypeError("w only support np.float32, float or Tensor type.")

        self.w = Parameter(initializer(w, [channel]), name='a')
        self.prelu = P.PReLU()
        self.relu = P.ReLU()
        self.assign = P.Assign()

    def construct(self, x):
        u = self.relu(self.w)
        v = self.prelu(x, u)
        if self.training:
            self.assign(self.w, u)
        return v


class HSwish(Cell):
    r"""
    rHard swish activation function.

    Applies hswish-type activation element-wise. The input is a Tensor with any valid shape.

    Hard swish is defined as:

    .. math::
        \text{hswish}(x_{i}) = x_{i} * \frac{ReLU6(x_{i} + 3)}{6},

    where :math:`x_{i}` is the :math:`i`-th slice along the given dim of the input Tensor.

    Inputs:
        - **input_data** (Tensor) - The input of HSwish.

    Outputs:
        Tensor, with the same type and shape as the `input_data`.

    Examples:
        >>> input_x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> hswish = nn.HSwish()
        >>> hswish(input_x)

    """

    def __init__(self):
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
        \text{hsigmoid}(x_{i}) = max(0, min(1, \frac{2 * x_{i} + 5}{10})),

    where :math:`x_{i}` is the :math:`i`-th slice along the given dim of the input Tensor.

    Inputs:
        - **input_data** (Tensor) - The input of HSigmoid.

    Outputs:
        Tensor, with the same type and shape as the `input_data`.

    Examples:
        >>> input_x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> hsigmoid = nn.HSigmoid()
        >>> hsigmoid(input_x)

    """

    def __init__(self):
        super(HSigmoid, self).__init__()
        self.hsigmoid = P.HSigmoid()

    def construct(self, x):
        return self.hsigmoid(x)


class LogSigmoid(Cell):
    r"""
    Logsigmoid activation function.

    Applies logsigmoid activation element-wise. The input is a Tensor with any valid shape.

    Logsigmoid is defined as:

    .. math::
        \text{logsigmoid}(x_{i}) = log(\frac{1}{1 + \exp(-x_i)}),

    where :math:`x_{i}` is the element of the input.

    Inputs:
        - **input_data** (Tensor) - The input of LogSigmoid.

    Outputs:
        Tensor, with the same type and shape as the `input_data`.

    Examples:
        >>> net = nn.LogSigmoid()
        >>> input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> logsigmoid = net(input_x)
        [-3.1326166e-01, -1.2692806e-01, -4.8587345e-02]

    """
    def __init__(self):
        super(LogSigmoid, self).__init__()
        self.mul = P.Mul()
        self.exp = P.Exp()
        self.add = P.TensorAdd()
        self.rec = P.Reciprocal()
        self.log = P.Log()

    def construct(self, input_x):
        neg_input = self.mul(input_x, -1)
        exp_neg_input = self.exp(neg_input)
        exp_neg_input_1 = self.add(exp_neg_input, 1)
        rec_exp_neg_input_1 = self.rec(exp_neg_input_1)
        ret = self.log(rec_exp_neg_input_1)
        return ret


_activation = {
    'softmax': Softmax,
    'logsoftmax': LogSoftmax,
    'relu': ReLU,
    'relu6': ReLU6,
    'tanh': Tanh,
    'gelu': GELU,
    'sigmoid': Sigmoid,
    'prelu': PReLU,
    'leakyrelu': LeakyReLU,
    'hswish': HSwish,
    'hsigmoid': HSigmoid,
    'logsigmoid': LogSigmoid,
}


def get_activation(name):
    """
    Gets the activation function.

    Args:
        name (str): The name of the activation function.

    Returns:
        Function, the activation function.

    Examples:
        >>> sigmoid = nn.get_activation('sigmoid')
    """
    if not name:
        return None

    if name not in _activation:
        raise KeyError("Unknown activation type")
    return _activation[name]()

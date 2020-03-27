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
"""pooling"""

from mindspore.ops import operations as P
from mindspore._checkparam import ParamValidator as validator
from mindspore._checkparam import Rel
from ..cell import Cell


class _PoolNd(Cell):
    """N-D  AvgPool"""

    def __init__(self,
                 kernel_size,
                 stride,
                 pad_mode,
                 padding=0,
                 pool=None):
        super(_PoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = validator.check_integer('padding', padding, 0, Rel.GE)
        self.pool = pool
        if self.pool is None:
            raise NotImplementedError

    def construct(self, x):
        return self.pool(x)

    def extend_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, pad_mode={pad_mode}'.format(**self.__dict__)


class MaxPool2d(_PoolNd):
    r"""
    Max pooling operation for temporal data.

    Applies a 2D max pooling over an input Tensor which can be regarded as a composition of 2D planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, MaxPool2d outputs
    regional maximum in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    Note:
        pad_mode for training only supports "same" and "valid".

    Args:
        kernel_size (int): Size of the window to take a max over.
        stride (int): Stride size of the window. Default: None.
        pad_mode (str): Select the mode of the pad. The optional values are
            "same" and "valid". Default: "valid".

            - same: Adopts the way of completion. Output height and width will be the same as
              the input. Total number of padding will be calculated for horizontal and vertical
              direction and evenly distributed to top and bottom, left and right if possible. Otherwise, the
              last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possibly largest height and width of output will be return
              without padding. Extra pixels will be discarded.
        padding (int): Now is not supported, mplicit zero padding to be added on both sides. Default: 0.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> pool = MaxPool2d(kernel_size=3, stride=1)
        >>> x = mindspore.Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
        [[[[1. 5. 5. 1.]
           [0. 3. 4. 8.]
           [4. 2. 7. 6.]
           [4. 9. 0. 1.]]
          [[3. 6. 2. 6.]
           [4. 4. 7. 8.]
           [0. 0. 4. 0.]
           [1. 8. 7. 0.]]]]
        >>> output = pool(x)
        >>> output.shape()
        (1, 2, 2, 2)
        >>> output
        [[[[7. 8.]
           [9. 9.]]
          [[7. 8.]
           [8. 8.]]]]
    """
    def __init__(self,
                 kernel_size=1,
                 stride=1,
                 pad_mode="VALID",
                 padding=0):
        max_pool = P.MaxPool(ksize=kernel_size,
                             strides=stride,
                             padding=pad_mode)
        self.is_autodiff_backend = False
        if self.is_autodiff_backend:

            # At present, pad mode of max pool is not unified, so it is a temporarily avoided
            pad_mode = validator.check_string('pad_mode', pad_mode.lower(), ['valid', 'same'])

            max_pool = P.MaxPoolWithArgmax(window=kernel_size,
                                           stride=stride,
                                           pad_mode=pad_mode,
                                           pad=padding)
        super(MaxPool2d, self).__init__(kernel_size, stride, pad_mode, padding, max_pool)

    def construct(self, x):
        if self.is_autodiff_backend:
            out = self.pool(x)[0]
        else:
            out = self.pool(x)
        return out


class AvgPool2d(_PoolNd):
    r"""
    Average pooling for temporal data.

    Applies a 2D average pooling over an input Tensor which can be regarded as a composition of 2D input planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, AvgPool2d outputs
    regional average in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \frac{1}{h_{ker} * w_{ker}} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    Note:
        pad_mode for training only supports "same" and "valid".

    Args:
        kernel_size (int): Size of the window to take a max over.
        stride (int): Stride size of the window. Default: None.
        pad_mode (str): Select the mode of the pad. The optional values are
            "same", "valid". Default: "valid".

            - same: Adopts the way of completion. Output height and width will be the same as
              the input. Total number of padding will be calculated for horizontal and vertical
              direction and evenly distributed to top and bottom, left and right if possible. Otherwise, the
              last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possibly largest height and width of output will be return
              without padding. Extra pixels will be discarded.
        padding (int): Now is not supported, implicit zero padding to be added on both sides. Default: 0.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> pool = AvgPool2d(kernel_size=3, stride=1)
        >>> x = mindspore.Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
        [[[[5. 5. 9. 9.]
            [8. 4. 3. 0.]
            [2. 7. 1. 2.]
            [1. 8. 3. 3.]]
           [[6. 8. 2. 4.]
            [3. 0. 2. 1.]
            [0. 8. 9. 7.]
            [2. 1. 4. 9.]]]]
        >>> output = pool(x)
        >>> output.shape()
        (1, 2, 2, 2)
        >>> output
        [[[[4.888889  4.4444447]
           [4.111111  3.4444444]]
          [[4.2222223 4.5555553]
           [3.2222223 4.5555553]]]]
    """
    def __init__(self,
                 kernel_size=1,
                 stride=1,
                 pad_mode="VALID",
                 padding=0):
        avg_pool = P.AvgPool(ksize=kernel_size,
                             strides=stride,
                             padding=pad_mode)
        super(AvgPool2d, self).__init__(kernel_size, stride, pad_mode, padding, avg_pool)

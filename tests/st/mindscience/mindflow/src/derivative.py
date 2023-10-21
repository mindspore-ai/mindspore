# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
derivative
"""
from mindspore import nn, ops, jacrev
from mindspore.ops import constexpr
from mindspore import dtype as mstype


class SimplifiedGradient(nn.Cell):
    """Simplify the results of the input network."""

    def __init__(self, net, order=1):
        super().__init__()
        if not isinstance(order, int):
            raise TypeError(
                "The type of order should be int, but got {}".format(type(order)))
        self.net = net
        self.axis = order - 1
        self.cast = ops.Cast()

    def construct(self, x):
        return self.cast(self.net(x).sum(axis=self.axis), mstype.float32)


@constexpr
def batched_jacobian(model):
    """
    Calculate Jacobian matrix of network model.

    Args:
        model (mindspore.nn.Cell): a network with the input dimension is in_channels and output dimension is
            out_channels.

    Returns:
        jacobian(Tensor), jacobi of the model. With the input dimension is [batch_size, in_channels], output dimension
        is [out_channels, batch_size, in_channels].

    Note:
        The version of MindSpore should be >= 2.0.0 for using `mindspore.jacrev`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn, ops, Tensor
        >>> from mindspore import dtype as mstype
        >>> from mindflow.operators import batched_jacobian
        >>> np.random.seed(123456)
        >>> class Net(nn.Cell):
        ...     def __init__(self, cin=2, cout=1, hidden=10):
        ...         super().__init__()
        ...         self.fc1 = nn.Dense(cin, hidden)
        ...         self.fc2 = nn.Dense(hidden, hidden)
        ...         self.fcout = nn.Dense(hidden, cout)
        ...         self.act = ops.Tanh()
        ...
        ...     def construct(self, x):
        ...         x = self.act(self.fc1(x))
        ...         x = self.act(self.fc2(x))
        ...         x = self.fcout(x)
        ...         return x
        >>> model = Net()
        >>> jacobian = batched_jacobian(model)
        >>> inputs = np.random.random(size=(3, 2))
        >>> res = jacobian(Tensor(inputs, mstype.float32))
        >>> print(res.shape)
        (1, 3, 2)
    """
    return jacrev(SimplifiedGradient(model, 1))


@constexpr
def batched_hessian(model):
    """
    Calculate Hessian matrix of network model.

    Args:
        model (mindspore.nn.Cell): a network with the input dimension is in_channels and output dimension is
            out_channels.

    Returns:
        hessian(Tensor), hessian of the model. With the input dimension is [batch_size, in_channels], output dimension
        is [out_channels, in_channels, batch_size, in_channels].

    Note:
        The version of MindSpore should be >= 2.0.0 for using `mindspore.jacrev`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn, ops, Tensor
        >>> from mindspore import dtype as mstype
        >>> from mindflow.operators import batched_hessian
        >>> np.random.seed(123456)
        >>> class Net(nn.Cell):
        ...     def __init__(self, cin=2, cout=1, hidden=10):
        ...         super().__init__()
        ...         self.fc1 = nn.Dense(cin, hidden)
        ...         self.fc2 = nn.Dense(hidden, hidden)
        ...         self.fcout = nn.Dense(hidden, cout)
        ...         self.act = ops.Tanh()
        ...
        ...     def construct(self, x):
        ...         x = self.act(self.fc1(x))
        ...         x = self.act(self.fc2(x))
        ...         x = self.fcout(x)
        ...         return x
        >>> model = Net()
        >>> hessian = batched_hessian(model)
        >>> inputs = np.random.random(size=(3, 2))
        >>> res = hessian(Tensor(inputs, mstype.float32))
        >>> print(res.shape)
        (1, 2, 3, 2)
    """
    return jacrev(SimplifiedGradient(batched_jacobian(model), 2))

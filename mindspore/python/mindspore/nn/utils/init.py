# Copyright 2024 Huawei Technologies Co., Ltd
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

"""init for nn.Cell."""
from __future__ import absolute_import

from contextlib import contextmanager
from mindspore.common.parameter import Parameter


@contextmanager
def no_init_parameters():
    r"""
    This interface is used to skip parameter initialization.

    In scenarios where a checkpoint is loaded, parameters within the network instantiation will be
    instantiated and occupy physical memory. Loading a checkpoint will replace the parameter values.
    Decorator can be applied during network instantiation to add an attribute `init_param` to all
    parameters within the current Cell, setting it to `init_param=False` .
    When `init_param=False` is detected, the initialization of the parameters is skipped,
    and the parameters are assigned values directly from the checkpoint during loading,
    which can optimize performance and reduce physical memory usage.

    .. warning::
        This interface is deprecated and will be removed after version 2.9.0.

    Note:
        Initialization of parameters created with `initializer` can only be skipped.
        Parameters created by `Tensor` or `numpy` cannot be skipped.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn, ops, load_checkpoint
        >>> from mindspore.common.initializer import initializer
        >>> from mindspore.nn.utils import no_init_parameters
        >>> # 1. Add a context manager to the network that requires delayed initialization
        >>> class Net(nn.Cell):
        ...     def __init__(self, in_channels, out_channels):
        ...         super().__init__()
        ...         self.weight = ms.Parameter(initializer("normal", [in_channels, out_channels], ms.float32))
        ...         self.bias = ms.Parameter(initializer("normal", [out_channels], ms.float32))
        ...         self.matmul = ops.MatMul()
        ...         self.add = ops.Add()
        ...
        ...     def construct(self, x):
        ...         x = self.matmul(x, self.weight)
        ...         x = self.add(x, self.bias)
        ...         return x
        >>> with no_init_parameters():
        ...     # After instantiation, all parameters in the net are not initialized
        ...     net = Net(28*28, 64)
        >>> # 2. Load checkpoint parameters to the net
        >>> load_checkpoint('./checkpoint/test_net.ckpt', net=net)
        >>> # 3. After loading the checkpoint, manually call init_parameters_data() to initialize
        >>> #    the uninitialized parameters in the net if needed. If the network is executed,
        >>> #    the framework will automatically call this interface.
        >>> net.init_parameters_data()
    """
    init_class = Parameter
    setattr(init_class, "init_param", False)
    try:
        yield
    finally:
        setattr(init_class, "init_param", True)

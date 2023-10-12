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

"""lazy_inline"""
from __future__ import absolute_import
import inspect
from functools import wraps


def lazy_inline(fn=None, attrs=None):
    """
    Make the cell to be reusable. The corresponding sub graph will not be inline at first.

    Registering the decorator of the built-in function `__init__` of a cell, the decorator
    will add the parameters of `__init__` according to the `attrs` as the attributes of this cell.

    .. warning::
        This feature is only supported on Ascend and is not supported on other hardwares.

    Args:
        fn (function): `__init__` function of a cell.
        attrs (Union[list[string], string]): The attributes list to add for the cell.

    Returns:
        function, original function.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.nn as nn
        >>> from mindspore import lazy_inline
        >>> from mindspore import context
        >>> from mindspore import ops
        >>> def conv3x3(in_channels, out_channels, stride=1, padding=1, pad_mode='pad'):
        ...     return nn.Conv2d(in_channels, out_channels,
        ...                      kernel_size=3, stride=stride, padding=padding, pad_mode=pad_mode)
        ...
        >>> def conv1x1(in_channels, out_channels, stride=1, padding=0, pad_mode='pad'):
        ...     return nn.Conv2d(in_channels, out_channels,
        ...                      kernel_size=1, stride=stride, padding=padding, pad_mode=pad_mode)
        ...
        >>> class Block(nn.Cell):
        ...     expansion = 4
        ...
        ...     @lazy_inline
        ...     def __init__(self,
        ...                  in_channels,
        ...                  out_channels,
        ...                  stride=1,
        ...                  down_sample=False):
        ...         super(Block, self).__init__()
        ...
        ...         out_chls = out_channels
        ...         self.conv1 = conv1x1(in_channels, out_chls, stride=1, padding=0)
        ...         self.bn1 = nn.BatchNorm2d(out_chls)
        ...
        ...         self.conv2 = conv3x3(out_chls, out_chls, stride=stride, padding=1)
        ...         self.bn2 = nn.BatchNorm2d(out_chls)
        ...
        ...         self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        ...         self.bn3 = nn.BatchNorm2d(out_channels)
        ...
        ...         self.relu = nn.ReLU()
        ...         self.downsample = down_sample
        ...
        ...         self.conv_down_sample = conv1x1(in_channels, out_channels,
        ...                                         stride=stride, padding=0)
        ...         self.bn_down_sample = nn.BatchNorm2d(out_channels)
        ...         self.add = ops.Add()
        ...
        ...     def construct(self, x):
        ...         identity = x
        ...
        ...         out = self.conv1(x)
        ...         out = self.bn1(out)
        ...         out = self.relu(out)
        ...
        ...         out = self.conv2(out)
        ...         out = self.bn2(out)
        ...         out = self.relu(out)
        ...
        ...         out = self.conv3(out)
        ...         out = self.bn3(out)
        ...
        ...         if self.downsample:
        ...             identity = self.conv_down_sample(identity)
        ...             identity = self.bn_down_sample(identity)
        ...
        ...         out = self.add(out, identity)
        ...         out = self.relu(out)
        ...
        ...         return out
        ...
        >>> class Net(nn.Cell):
        ...     def __init__(self, block, num_classes=100):
        ...         super(Net, self).__init__()
        ...
        ...         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
        ...         self.bn1 = nn.BatchNorm2d(64)
        ...         self.relu = nn.ReLU()
        ...         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        ...
        ...         self.layer = self.MakeLayer(
        ...             block, 50, in_channels=64, out_channels=2048, stride=2)
        ...         self.avgpool = nn.AvgPool2d(7, 1)
        ...         self.flatten = ops.Flatten()
        ...
        ...     def MakeLayer(self, block, layer_num, in_channels, out_channels, stride):
        ...         layers = []
        ...         resblk = block(in_channels, out_channels,
        ...                        stride=stride, down_sample=True)
        ...         layers.append(resblk)
        ...
        ...         for _ in range(1, layer_num):
        ...             resblk = block(out_channels, out_channels, stride=1)
        ...             layers.append(resblk)
        ...
        ...         return nn.SequentialCell(layers)
        ...
        ...     def construct(self, x):
        ...         x = self.conv1(x)
        ...         x = self.bn1(x)
        ...         x = self.relu(x)
        ...         x = self.maxpool(x)
        ...         x = self.layer(x)
        ...         x = self.avgpool(x)
        ...         x = self.flatten(x)
        ...         return x
        ...
        >>> def test_compile():
        ...     net = Net(Block)
        ...     inp = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
        ...     net(inp)
        ...
        >>> context.set_context(mode=context.GRAPH_MODE,
        ...                     save_graphs=True, save_graphs_path="./lazy")
        ...
        >>> test_compile()
    """

    def wrap_cell(fn):
        @wraps(fn)
        def deco(self, *args, **kwargs):
            arguments = []
            if attrs is None:
                bound_args = inspect.signature(fn).bind(self, *args, **kwargs)
                arguments = bound_args.arguments
                del arguments['self']
                arguments = arguments.values()
            fn(self, *args, **kwargs)
            if attrs is None:
                self.cell_init_args = "lazy_inline_" + type(self).__name__ + str(arguments)
                return

            if isinstance(attrs, list):
                for item in attrs:
                    if not isinstance(item, str):
                        raise ValueError(f"attr must be a string")
                    if hasattr(self, item):
                        arguments.append(getattr(self, item))
            elif isinstance(attrs, str):
                if hasattr(self, attrs):
                    arguments = getattr(self, attrs)
            else:
                raise ValueError(f"attrs must be list or string")
            self.cell_init_args = "lazy_inline_" + type(self).__name__ + str(arguments)

        return deco

    if fn is not None:
        return wrap_cell(fn)
    return wrap_cell

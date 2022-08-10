# Copyright 2022 Huawei Technologies Co., Ltd
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
BatchReadWrite
"""
from __future__ import absolute_import

from mindspore.nn.cell import Cell
from mindspore.ops.operations._rl_inner_ops import BatchAssign


class BatchWrite(Cell):
    r"""BatchWrite: write a list of parameters to assign the target.

    .. warning::
        This is an experiential prototype that is subject to change and/or deletion.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import nn
        >>> from mindspore.common.parameter import Parameter, ParameterTuple
        >>> from mindspore.nn.reinforcement import BatchWrite
        >>> class SourceNet(nn.Cell):
        ...   def __init__(self):
        ...     super(SourceNet, self).__init__()
        ...     self.a = Parameter(Tensor(0.5, mstype.float32), name="a")
        ...     self.dense = nn.Dense(in_channels=16, out_channels=1, weight_init=0)
        >>> class DstNet(nn.Cell):
        ...   def __init__(self):
        ...     super(DstNet, self).__init__()
        ...     self.a = Parameter(Tensor(0.1, mstype.float32), name="a")
        ...     self.dense = nn.Dense(in_channels=16, out_channels=1)
        >>> class Write(nn.Cell):
        ...   def __init__(self, dst, src):
        ...     super(Write, self).__init__()
        ...     self.w = BatchWrite()
        ...     self.dst = ParameterTuple(dst.trainable_params())
        ...     self.src = ParameterTuple(src.trainable_params())
        ...   def construct(self):
        ...     success = self.w(self.dst, self.src)
        ...     return success
        >>> dst_net = DstNet()
        >>> source_net = SourceNet()
        >>> nets = nn.CellList()
        >>> nets.append(dst_net)
        >>> nets.append(source_net)
        >>> success = Write(nets[0], nets[1])()
    """
    def __init__(self):
        """Initialize BatchWrite"""
        super(BatchWrite, self).__init__()
        self.write = BatchAssign(lock=True)

    def construct(self, dst, src):
        """
        Write the source parameter list to assign the dst.

        Inputs:
            - **dst** (tuple) - A paramameter tuple of the dst model.
            - **src** (tuple) - A paramameter tuple of the source model.

        Returns:
            Bool, true.
        """
        self.write(dst, src)
        return True


class BatchRead(Cell):
    r"""BatchRead: read a list of parameters to assign the target.

    .. warning::
        This is an experiential prototype that is subject to change and/or deletion.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import nn
        >>> from mindspore.common.parameter import Parameter, ParameterTuple
        >>> from mindspore.nn.reinforcement import BatchRead
        >>> class SNet(nn.Cell):
        ...   def __init__(self):
        ...     super(SNet, self).__init__()
        ...     self.a = Parameter(Tensor(0.5, mstype.float32), name="a")
        ...     self.dense = nn.Dense(in_channels=16, out_channels=1, weight_init=0)
        >>> class DNet(nn.Cell):
        ...   def __init__(self):
        ...     super(DNet, self).__init__()
        ...     self.a = Parameter(Tensor(0.1, mstype.float32), name="a")
        ...     self.dense = nn.Dense(in_channels=16, out_channels=1)
        >>> class Read(nn.Cell):
        ...   def __init__(self, dst, src):
        ...     super(Read, self).__init__()
        ...     self.read = BatchRead()
        ...     self.dst = ParameterTuple(dst.trainable_params())
        ...     self.src = ParameterTuple(src.trainable_params())
        ...   def construct(self):
        ...     success = self.read(self.dst, self.src)
        ...     return success
        >>> dst_net = DNet()
        >>> source_net = SNet()
        >>> nets = nn.CellList()
        >>> nets.append(dst_net)
        >>> nets.append(source_net)
        >>> success = Read(nets[0], nets[1])()

    """
    def __init__(self):
        """Initialize BatchRead"""
        super(BatchRead, self).__init__()
        self.read = BatchAssign(lock=False)

    def construct(self, dst, src):
        """
        Read the source parameter list to assign the dst.

        Inputs:
            - **dst** (tuple) - A paramameter tuple of the dst model.
            - **src** (tuple) - A paramameter tuple of the source model.

        Returns:
            Bool, true.
        """
        self.read(dst, src)
        return True

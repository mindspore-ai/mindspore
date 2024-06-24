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
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.nn.reinforcement._batch_read_write import BatchRead, BatchWrite

from tests.mark_utils import arg_mark


class DstNet(nn.Cell):
    '''Dst net'''
    def __init__(self):
        super(DstNet, self).__init__()
        self.a = Parameter(Tensor(0.1, mstype.float32), name="a")
        self.dense = nn.Dense(in_channels=16, out_channels=1)

    def construct(self, data):
        d = self.dense(data)
        out = d + self.a
        return out


class SourceNet(nn.Cell):
    '''Source net'''
    def __init__(self):
        super(SourceNet, self).__init__()
        self.a = Parameter(Tensor(0.5, mstype.float32), name="a")
        self.dense = nn.Dense(in_channels=16, out_channels=1, weight_init=0)

    def construct(self, data):
        d = self.dense(data)
        out = d + self.a
        return out


class Write(nn.Cell):
    '''Write cell'''
    def __init__(self, dst, src):
        super(Write, self).__init__()
        self.write = BatchWrite()
        self.dst = ParameterTuple(dst.trainable_params())
        self.src = ParameterTuple(src.trainable_params())

    def construct(self):
        success = self.write(self.dst, self.src)
        return success


class Read(nn.Cell):
    '''Read cell'''
    def __init__(self, dst, src):
        super(Read, self).__init__()
        self.read = BatchRead()
        self.dst = ParameterTuple(dst.trainable_params())
        self.src = ParameterTuple(src.trainable_params())

    def construct(self):
        success = self.read(self.dst, self.src)
        return success


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_read_write_model_gpu():
    """
    Feature: BatchPushPull gpu TEST.
    Description: Test the batch assign.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dst_net = DstNet()
    source_net = SourceNet()
    dst_param = dst_net.trainable_params()
    source_param = source_net.trainable_params()
    nets = nn.CellList()
    nets.append(dst_net)
    nets.append(source_net)
    # Test read source net's params to replace dst_net's params.
    _ = Read(nets[0], nets[1])()
    assert np.allclose(dst_param[0].asnumpy(), 0.5)

    # Test write dst net's params to overwrite the source.
    dst_net2 = DstNet()
    nets[0] = dst_net2
    _ = Write(nets[1], nets[0])()
    assert np.allclose(source_param[0].asnumpy(), 0.1)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_read_write_model_cpu():
    """
    Feature: BatchPushPull cpu TEST.
    Description: Test the batch assign.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dst_net = DstNet()
    source_net = SourceNet()
    dst_param = dst_net.trainable_params()
    source_param = source_net.trainable_params()
    cpu_nets = nn.CellList()
    cpu_nets.append(dst_net)
    cpu_nets.append(source_net)
    _ = Read(cpu_nets[0], cpu_nets[1])()
    assert np.allclose(dst_param[0].asnumpy(), 0.5)

    dst_net2 = DstNet()
    cpu_nets[0] = dst_net2
    _ = Write(cpu_nets[1], cpu_nets[0])()
    assert np.allclose(source_param[0].asnumpy(), 0.1)

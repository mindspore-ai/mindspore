# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
""" test control ops """
import numpy as np
import pytest

from mindspore import Tensor
from mindspore import nn, context
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.common.parameter import Parameter, ParameterTuple

grad_by_list = C.GradOperation(get_by_list=True)
grad_all = C.GradOperation(get_all=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_switch_layer_with_single_prim():
    """
    Feature: SwitchLayer
    Description: run switch layer case
    Expectation: success.
    """
    class SwitchLayerCell(nn.Cell):
        def __init__(self):
            super(SwitchLayerCell, self).__init__()
            self.layers = (nn.ReLU(), nn.ReLU())
            self.z3 = Parameter(
                Tensor(np.full([128, 96], 0.6, dtype=np.float32)), name='z3')

        def construct(self, index, x):
            ret = self.layers[index](x) * self.z3
            return ret

    index = Tensor(0, dtype=mstype.int32)
    net = SwitchLayerCell()
    net(index, Tensor(np.full([128, 96], 0.6, dtype=np.float32)))
    grad_by_list(net, ParameterTuple(net.trainable_params()))(index,
                                                              Tensor(np.full([128, 96], 0.6, dtype=np.float32)))
    grad_all(net)(index, Tensor(np.full([128, 96], 0.6, dtype=np.float32)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_switch_layer_with_index_out_of_range():
    """
    Feature: SwitchLayer
    Description: run switch layer case
    Expectation: catch exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.layers = (nn.ReLU(),)

        def construct(self, inputs, index):
            inputs = self.layers[index](inputs)
            return inputs

    context.set_context(mode=context.GRAPH_MODE)
    inputs = Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)).astype(np.float32), mstype.float32)
    index = Tensor(-2, mstype.int32)
    net = Net()
    with pytest.raises(IndexError) as err:
        net(inputs, index)
    assert "Given index -2 out of range" in str(err.value)

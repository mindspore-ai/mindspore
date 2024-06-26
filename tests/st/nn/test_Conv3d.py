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

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops

from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(4, 3, 3), dtype=ms.float16)

    def construct(self, x):
        out = self.conv3d(x)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_conv3d_para_customed_dtype(mode):
    """
    Feature: Conv3d
    Description: Verify the result of Conv3d specifying customed para dtype.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = Tensor(np.ones([16, 3, 10, 32, 32]), ms.float16)
    output = net(x)
    expect_output_shape = (16, 32, 10, 32, 32)
    assert np.allclose(expect_output_shape, output.shape)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@test_utils.run_test_with_On
def test_conv3d_input_5d():
    """
    Feature: Conv3d 5d input
    Description: Verify the result of Conv3d 5d input.
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE, ascend_config={"precision_mode": "force_fp16"})
    class Network(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()
            self.conv1 = nn.Conv3d(1, 1, kernel_size=5, pad_mode="same", padding=0, has_bias=False, weight_init="One")
            self.reducemin = ops.ReduceMin(keep_dims=True)
            self.reducesum = ops.ReduceSum(keep_dims=True)
            self.add = ops.Add()
            self.square = ops.Square()
            self.abs = ops.Abs()
            self.concat = ops.Concat()
            self.batchnorm = nn.BatchNorm3d(5)

        def construct(self, data1, data2):
            batchnorm3d_01 = self.batchnorm(data1)
            batchnorm3d_02 = self.batchnorm(data1)
            reducesum_01 = self.reducesum(batchnorm3d_02, 1)
            add_01 = self.add(reducesum_01, data2)
            reducemin_01 = self.reducemin(add_01, 1)
            relu_01 = self.relu(batchnorm3d_01)
            abs_01 = self.abs(relu_01)
            square_01 = self.square(abs_01)
            reducemin_02 = self.reducemin(square_01, 1)
            concat_01 = self.concat((reducemin_02, reducemin_01))
            conv_01 = self.conv1(concat_01)
            relu_03 = self.relu(conv_01)
            output = relu_03
            return  output

    data1 = Tensor(np.ones([1, 5, 5, 5, 4]).astype(np.float32))
    data2 = Tensor(np.ones([1, 5, 5, 4]).astype(np.float32))

    ms.set_context(device_target="CPU")
    cpu_mode = Network()
    cpu_out = cpu_mode(data1, data2).asnumpy()

    ms.set_context(device_target="Ascend")
    npu_mode = Network()
    npu_out = npu_mode(data1, data2).asnumpy()

    assert np.allclose(cpu_out, npu_out, 0.001, 0.001)

# Copyright 2025 Huawei Technologies Co., Ltd
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


import os
import time
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, nn, Parameter
from mindspore.common.initializer import initializer
from tests.mark_utils import arg_mark


class TestNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(initializer('ones', (64, 64), ms.float16))
        self.matmul = ops.MatMul(False, False)

    def construct(self, x):
        return self.matmul(x, self.weight)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
        card_mark='onecard', essential_mark='unessential', mem_peak=4.01)
def test_ptq_offload_pattern():
    """
    Feature: test offload mode in ptq algorithm.
    Description: test offload mode in ptq algorithm.
    Expectation: finish without exception and error.
    """
    ms.set_device('Ascend')
    ms.set_context(mode=ms.PYNATIVE_MODE)

    net = TestNet()
    x = Tensor(np.random.randn(64, 64), ms.float16)
    net(x).asnumpy()
    net.weight._offload()
    qweight = Parameter(initializer('ones', (64, 64), ms.int8))
    tmp_weight = net.weight + 1
    tmp_weight = net.weight.astype(ms.int8)
    tmp_weight = tmp_weight.reshape(qweight.shape)
    tmp_weight = tmp_weight.transpose((1, 0))
    tmp_weight = tmp_weight.squeeze()
    qweight.set_data(tmp_weight)
    net.weight._offload()
    del net.weight
    net.weight = qweight
    net.weight._offload()
    ckpt_path = './test.ckpt'
    ms.save_checkpoint(net.parameters_dict(), ckpt_path)
    time.sleep(5)
    assert os.path.exists(ckpt_path)
    os.remove(ckpt_path)

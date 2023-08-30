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
import mindspore as ms
import numpy as np

from mindspore import ops
from mindspore import nn
from mindspore.ops import auto_generate as ops


class AvgPoolNet(nn.Cell):
    def __init__(self):
        super(AvgPoolNet, self).__init__()
        self.avg_pool = ops.AvgPool(kernel_size=2, strides=1, pad_mode="VALID", data_format="NCHW")

    @ms.jit
    def construct(self, x):
        return self.avg_pool(x)


def test_avg_pool_oop():
    ms.context.set_context(precompile_only=True)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    net = AvgPoolNet()
    out = net(x)
    print("out:", out)


def test_avg_pool_func():
    pass

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
import numpy as np
import pytest

import mindspore.context as context
from mindspore import Tensor
import mindspore.nn as nn
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, axis=0, out_nums=1):
        super(Net, self).__init__()
        self.split = P.Split(axis, out_nums)

    def construct(self, x):
        return self.split(x)


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_split():
    x = np.array([[[1, -1, 1], [2, -2, 2]],
                  [[3, -3, 3], [4, -4, 4]],
                  [[5, -5, 5], [6, -6, 6]]]).astype(np.float32)

    split_op = Net(0, 3)
    outputs = split_op(Tensor(x))
    for i, out in enumerate(outputs):
        assert (out.asnumpy() == x[i]).all()


def test_split_4d():
    x_np = np.random.randn(2, 6, 4, 4).astype(np.float32)
    y = np.split(x_np, 3, axis=1)

    split_op = Net(1, 3)
    outputs = split_op(Tensor(x_np))

    for i, out in enumerate(outputs):
        assert (out.asnumpy() == y[i]).all()

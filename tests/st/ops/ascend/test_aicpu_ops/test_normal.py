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
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common import Tensor
from mindspore.common import dtype as mstype


context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, shape=None, mean=0.0, stddev=1.0, seed=0):
        super(Net, self).__init__()
        self._mean = Tensor(mean, mstype.float32)
        self._stddev = Tensor(stddev, mstype.float32)
        self._normal = P.Normal(seed=seed)
        self._shape = shape

    def construct(self):
        return self._normal(self._shape, self._mean, self._stddev)


def test_net_3x2x4():
    mean = 0.0
    stddev = 1.0
    seed = 0
    net = Net((3, 2, 4), mean, stddev, seed)
    out = net()
    assert out.shape == (3, 2, 4)

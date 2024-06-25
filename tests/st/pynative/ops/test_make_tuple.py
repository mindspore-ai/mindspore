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
""" test_make_tuple """
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from tests.mark_utils import arg_mark


# pylint: disable=unused-argument
def setup_module(module):
    ms.set_context(mode=ms.PYNATIVE_MODE)


class Net(nn.Cell):
    def construct(self, x, y):
        return ms.ops.make_tuple(x, y)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='essential')
def test_make_tuple():
    """
    Feature: Unify the dynamic mode and the static mode.
    Description: Support ops.make_tuple in PyNative mode.
    Expectation: No exception.
    """
    x = ms.Tensor(np.ones([1, 5, 10, 10], dtype=np.float32))
    y = ms.Tensor(np.ones([2, 10, 20, 20], dtype=np.float32))
    net = Net()
    net(x, y)

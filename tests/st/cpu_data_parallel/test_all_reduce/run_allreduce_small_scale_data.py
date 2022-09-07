# Copyright 2021 Huawei Technologies Co., Ltd
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

"""define and run AllReduce network"""

import numpy as np

from mindspore import Tensor
from mindspore import context
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.communication.management import init, get_group_size

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
context.set_ps_context(enable_ssl=False)
init()
context.set_auto_parallel_context(parallel_mode="data_parallel", gradients_mean=True, device_num=get_group_size())


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.all_reduce = P.AllReduce().add_prim_attr("fusion", 1)

    def construct(self, x, y, z):
        t = (x * x, y * y, z * z)
        n_t = ()
        for i in t:
            n_t += (self.all_reduce(i),)
        return n_t


def run_all_reduce():
    """ Run all reduce"""
    all_reduce = Net()
    x_np = np.arange(2).reshape((2, 1)).astype(np.float32)
    y_np = np.arange(3).reshape((3, 1)).astype(np.float32)
    z_np = np.arange(2).reshape((2, 1)).astype(np.float32)
    x_input = Tensor(x_np)
    y_input = Tensor(y_np)
    z_input = Tensor(z_np)
    output = all_reduce(x_input, y_input, z_input)
    assert np.array_equal(output[0].asnumpy(), x_input * x_input * get_group_size())
    assert np.array_equal(output[1].asnumpy(), y_input * y_input * get_group_size())
    assert np.array_equal(output[2].asnumpy(), z_input * z_input * get_group_size())


run_all_reduce()

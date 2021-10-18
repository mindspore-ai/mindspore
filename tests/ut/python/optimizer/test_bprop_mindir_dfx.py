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
"""DFX test for bprop mindir"""

import os
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore.ops._grad as g


class Net(nn.Cell):
    def __init__(self, op):
        super(Net, self).__init__()
        self.op = op

    def construct(self, *inputs):
        return self.op(*inputs)


class GradNet(nn.Cell):
    def __init__(self, network):
        super(GradNet, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, *inputs):
        gout = self.grad(self.network)(*inputs)
        return gout


def test_remove_bprop_fle():
    """
    Feature: Bprop pre-compilation.
    Description: Remove a bprop file, compile a grad net with a bprop not defined in this file.
    Expectation: The grad net can be complied successfully.
    """
    bprop_path = g.__file__
    bprop_installed_dir = bprop_path[: bprop_path.rindex('/')]
    nn_bprop_path = bprop_installed_dir + '/grad_nn_ops.py'
    new_path = bprop_installed_dir + '/new'
    os.mkdir(new_path)
    new_nn_bprop_path = bprop_installed_dir + '/new/grad_nn_ops.py'
    os.rename(nn_bprop_path, new_nn_bprop_path)
    x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
    ones_like = Net(P.OnesLike())
    grad = GradNet(ones_like)
    grad.compile(x)
    os.rename(new_nn_bprop_path, nn_bprop_path)
    os.rmdir(new_path)

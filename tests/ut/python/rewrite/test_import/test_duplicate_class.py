# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import nn
from mindspore import Tensor
from mindspore import rewrite
from .top_net import TopNet
import numpy as np


class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.top_net = TopNet()

    def construct(self, x):
        x = x + Tensor([1, 2, 3])
        x = self.top_net(x)
        return x

def test_import_duplicated_name_class():
    """
    Feature: Test import duplicate name class.
    Description: Import and use a custom class with duplicate name, here is 'Tensor'.
    Expectation: Name of 'Tensor' in TopNet is modified to 'Tensor_1'.
    """
    my_net = MyNet()
    y0 = my_net(Tensor([1, 2, 3]))
    stree = rewrite.SymbolTree.create(my_net)
    codes = stree.get_code()
    assert codes.count("tmp = Tensor_1()") == 1
    new_net = stree.get_network()
    y1 = new_net(Tensor([1, 2, 3]))
    assert np.allclose(y0.asnumpy(), y1.asnumpy())

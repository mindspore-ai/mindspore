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
"""test import"""
from __future__ import absolute_import
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.nn as nn  # pylint: disable=W0404
from mindspore.rewrite import SymbolTree
from .net.NetUtImport import NetUtImport
try:
    from mindspore import Tensor
except ImportError:
    import mindspore.Tensor as Tensor


class MyNetUtImport(nn.Cell):
    def __init__(self):
        super().__init__()
        self.blocks = nn.CellList()
        for _ in range(2):
            block = NetUtImport()
            self.blocks.append(block)

    def construct(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_import():
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with relative imports in subnet class define file.
    Expectation: Success.
    """
    x = Tensor(np.ones([1, 1, 32, 32]), mindspore.float32)
    net = MyNetUtImport()
    stree = SymbolTree.create(net)
    new_net = stree.get_network()
    y0 = net(x)
    y = new_net(x)
    assert Tensor.equal(y0, y).all()
    codes = stree.get_code()
    # the priority of class with the same name:  defined class in the current module
    # > explicitly imported class from other modules
    assert codes.count("class MyNetUtImportOpt(MyNetUtImport, nn.Cell):") == 1, codes
    assert codes.count("class NetUtImportOpt(NetUtImport, FatherNetOpt):") == 1, codes
    assert codes.count("class SubNetUtImportOpt(SubNetUtImport, FatherNetOpt):") == 1, codes
    assert codes.count("class NetBUtImportOpt(NetBUtImport, nn.Cell):") == 1, codes
    assert codes.count("class FatherNetOpt(FatherNet, nn.Cell):") == 1, codes

    # duplicated modules process
    assert codes.count("import mindspore.nn as nn") == 1

    try_import_statement = '''
try:
    from mindspore import Tensor
except ImportError:
    import mindspore.Tensor as Tensor
'''
    # import in try statement process
    assert codes.count(try_import_statement) == 1

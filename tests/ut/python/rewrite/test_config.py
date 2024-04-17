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
"""test for rewrite configurations."""
from mindspore import nn, rewrite


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2dTranspose(3, 5, 4)

    def construct(self, x):
        x = self.conv(x)
        return x

def test_clear_caches():
    """
    Feature: clear caches
    Description: test caches can be cleared.
    Expectation: caches can be cleared.
    """
    net = Net()
    # pylint: disable=protected-access
    rewrite.common.namespace._ms_cells_to_subtree = True
    # pylint: disable=protected-access
    rewrite.parsers.AssignParser._share_one_implementation = True
    stree = rewrite.SymbolTree.create(net)
    stree.get_network()
    # pylint: disable=protected-access
    rewrite.parsers.AssignParser._share_one_implementation = False
    # pylint: disable=protected-access
    rewrite.common.namespace._ms_cells_to_subtree = False
    # pylint: disable=protected-access
    assert len(rewrite.parsers.AssignParser._cached_trees) == 1
    # pylint: disable=protected-access
    assert len(rewrite.parsers.AssignParser._cached_functions) == 1
    # pylint: disable=protected-access, len-as-condition
    assert len(rewrite.parsers.AssignParser._cached_cell_containers) == 0
    rewrite.common.config.clear_caches()
    # pylint: disable=protected-access, len-as-condition
    assert len(rewrite.parsers.AssignParser._cached_trees) == 0
    # pylint: disable=protected-access, len-as-condition
    assert len(rewrite.parsers.AssignParser._cached_functions) == 0
    # pylint: disable=protected-access, len-as-condition
    assert len(rewrite.parsers.AssignParser._cached_cell_containers) == 0

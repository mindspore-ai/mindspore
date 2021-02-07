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

import mindspore.context as context
import mindspore.nn as nn

context.set_context(mode=context.GRAPH_MODE)
recompute_prefix = 'recompute_'


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, input_x):
        output = self.pool(input_x)
        return output


def test_set_recompute_true():
    net = Net()
    net.pool.recompute()
    assert net.pool.get_scope() == recompute_prefix


def test_set_recompute_false():
    net = Net()
    net.pool.recompute(False)
    assert net.pool.get_scope() is None


def test_set_recompute_true_twice():
    net = Net()
    net.pool.recompute()
    net.pool.recompute()
    assert net.pool.get_scope() == recompute_prefix


def test_set_recompute_false_twice():
    net = Net()
    net.pool.recompute(False)
    net.pool.recompute(False)
    assert net.pool.get_scope() is None


def test_reset_recompute1():
    net = Net()
    net.pool.recompute(True)
    net.pool.recompute(False)
    assert net.pool.get_scope() == ""


def test_reset_recompute2():
    net = Net()
    net.pool.recompute(False)
    net.pool.recompute(True)
    assert net.pool.get_scope() == recompute_prefix


def test_set_scope_and_set_recompute_repeatedly():
    net = Net()
    net.pool.recompute(True)
    assert net.pool.get_scope() == recompute_prefix
    net.pool.recompute(False)
    assert net.pool.get_scope() == ""
    net.pool.recompute(True)
    assert net.pool.get_scope() == recompute_prefix
    net.pool.recompute(False)
    assert net.pool.get_scope() == ""

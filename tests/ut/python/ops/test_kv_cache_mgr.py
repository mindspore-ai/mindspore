# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
""" test array ops """
import numpy as np
import mindspore as ms
import mindspore.context as context
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops.operations.inner_ops import KVCacheMgr


class Net(Cell):
    def __init__(self, B, H, S, D):
        super().__init__()
        self.past = Parameter(Tensor(np.zeros((B, H, S, D)), ms.float16), name="past")
        self.kv_cache_mgr = KVCacheMgr()

    def construct(self, curr, index):
        past = self.kv_cache_mgr(self.past, curr, index)
        return past + past


def test_kv_cache_mgr():
    """
    Feature: KVCacheMgr operator.
    Description: KVCacheMgr test.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, save_graphs_path='./ir')
    B, H, S, D = 8, 8, 1024, 128
    curr = Tensor(np.ones((B, H, 1, D)), ms.float16)
    index = Tensor([1, 2, 3, 4, 5, 6, 7, 8], ms.int32)
    net = Net(B, H, S, D)
    ret = net(curr, index)
    print(ret.asnumpy())

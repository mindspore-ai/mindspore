# Copyright 2025 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

''' test Tensor setitem side-effect'''

import mindspore
from mindspore import jit, nn, ops, Tensor
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_executed_by_graph_mode, assert_equal


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_kvcache_tensor_setitem_v1():
    """
    Feature: Tensor setitem side-effect.
    Description: Tensor setitem by slice.
    Expectation: No graph break, side-effect restored correctly.
    """

    class Attention(nn.Cell):
        def __init__(self):
            super().__init__()
            self.kvcache = ops.zeros((2, 3, 4), dtype=mindspore.float32)

        def construct(self, x: Tensor, start_pos: int) -> Tensor:
            B = x.shape[0]
            seq_len = x.shape[1]
            end_pos = start_pos + seq_len
            kv = x * 2
            self.kvcache[:B, start_pos:end_pos] = self.norm(kv)
            return kv

        def norm(self, x: Tensor) -> Tensor:
            return x + 1

    class Model(nn.Cell):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

        def construct(self, x: Tensor, start_pos: int) -> Tensor:
            return x + self.attn(x, start_pos)

    model = Model()
    x = ops.randn((2, 3, 4), dtype=mindspore.float32)
    o1 = model(x, 0)

    compiled_model = Model()
    compiled_model.construct = jit(compiled_model.construct, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_model(x, 0)

    assert_equal(o1, o2)
    assert_equal(model.attn.kvcache, compiled_model.attn.kvcache)
    assert_executed_by_graph_mode(compiled_model.construct)

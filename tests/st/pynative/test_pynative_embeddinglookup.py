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
""" test_pynative_embeddinglookup """
import pytest
import numpy as np
import mindspore.ops.operations as op
from mindspore import Tensor, context
from mindspore.nn import Cell

def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

class MetaFactory:
    def __init__(self):
        self.device_target = context.get_context('device_target')
        self.rank_size = None
        self.device_id = None
        self.global_rank_id = None

class OpsFactory(MetaFactory):
    def __init__(self, dtype=np.float16):
        super().__init__()
        self.dtype = dtype
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype == np.float32:
            self.loss = 1e-4
        elif self.dtype == np.float64:
            self.loss = 1e-5
        else:
            self.loss = 0

class EmbeddingLookup(Cell):
    def __init__(self, offset):
        super().__init__()
        self.op = op.EmbeddingLookup()
        self.offset = offset

    def construct(self, params, indices):
        x = self.op(params, indices, self.offset)
        return x

class EmbeddingLookupFactory(OpsFactory):
    def __init__(self, params_shape, indices_shape, offset=0, low=0, high=2, dtype=np.float32, ids_type=np.int32):
        super().__init__(dtype=dtype)
        self.input_np = np.random.randn(*params_shape).astype(dtype)
        self.indices_np = np.random.randint(low, high, size=indices_shape).astype(ids_type)
        self.offset = offset
        self.output_grad_np = None

    def forward_mindspore_impl(self):
        net = EmbeddingLookup(self.offset)
        out = net(Tensor(self.input_np), Tensor(self.indices_np))
        return out.asnumpy()

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_embeddinglookup_indices_outrange():
    fact = EmbeddingLookupFactory(params_shape=(2, 4), indices_shape=(2, 3), low=1, high=3, offset=10, dtype=np.int8)
    out = fact.forward_mindspore_impl()
    out_expect = np.zeros((2, 3, 4))
    np.allclose(out_expect, out)

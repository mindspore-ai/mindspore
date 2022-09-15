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

"""test embedding_lookup dynamic shape"""

import numpy as np
import pytest
from mindspore import context
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.embedding = P.EmbeddingLookup().add_prim_attr("primitive_target", "CPU")
        self.offset = 4

    def construct(self, param, index):
        return self.embedding(param, index, self.offset)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_embedding_look_up0():
    """
    Feature: test embedding_lookup op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    params = Tensor(
        np.array([[8, 9], [10, 11], [12, 13], [14, 15]]), mstype.float32)
    indices = Tensor(np.array([5, 2, 8, 5]), mstype.int32)
    params_dyn = Tensor(shape=[None, None], dtype=params.dtype)
    indices_dyn = Tensor(shape=[None], dtype=indices.dtype)
    embedding = Net()
    embedding.set_inputs(params_dyn, indices_dyn)
    out = embedding(params, indices)
    expect_shape = (4, 2)
    assert out.asnumpy().shape == expect_shape

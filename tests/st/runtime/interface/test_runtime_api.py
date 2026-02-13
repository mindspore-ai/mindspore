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
import numpy as np
from tests.mark_utils import arg_mark
import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore._c_expression import current_stream_id as current_stream_id_
from mindspore.graph.api import _bind_device_context


class Add(nn.Cell):
    def __init__(self):
        super(Add, self).__init__()
        self.add = P.Add()

    def construct(self, x1, x2):
        return self.add(x1, x2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bind_cuda_ctx_api():
    """
    Feature: _bind_device_ctx api
    Description: Test _bind_device_ctx api.
    Expectation: The results are as expected
    """
    x1 = Tensor(np.array([1]))
    x2 = Tensor(np.array([2]))

    _bind_device_context()
    net = Add()
    output = net(x1, x2)
    assert output.asnumpy() == np.array([3])

    _bind_device_context()
    output = net(x1, output)
    assert output.asnumpy() == np.array([4])


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
    mem_peak=4.00,
)
def test_current_stream_id():
    """
    Feature: current_stream_id
    Description: Test current_stream_id api.
    Expectation: The results are as expected
    """
    assert current_stream_id_() == 0
    stream = mindspore.runtime.Stream(stream_id=1)
    with mindspore.runtime.StreamCtx(stream):
        assert current_stream_id_() == 1
    assert current_stream_id_() == 0

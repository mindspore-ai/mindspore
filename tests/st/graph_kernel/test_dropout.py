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
import numpy as np
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, keep_prob):
        super(Net, self).__init__()
        self.drop = P.Dropout(keep_prob)

    def construct(self, x_):
        return self.drop(x_)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_dropout():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    x_shape = [4096, 768]
    x = np.ones(x_shape).astype(np.float32)
    keep_prob = 0.9
    dropout = Net(keep_prob)
    tx = Tensor(x)
    output, mask = dropout(tx)

    output_np = output.asnumpy()
    elem_count = x.size
    nonzero_count = np.count_nonzero(output_np)
    assert (elem_count * (keep_prob - 0.1)) < nonzero_count < (elem_count * (keep_prob + 0.1))
    output_sum = np.sum(output_np)
    x_sum = np.sum(x)
    assert abs(output_sum - x_sum) / x_sum < 0.1
    # check mask
    mask_np = mask.asnumpy()
    mask_sum = np.sum(mask_np)
    assert np.count_nonzero(mask_np) == nonzero_count
    assert abs(mask_sum - nonzero_count) / nonzero_count < 0.1

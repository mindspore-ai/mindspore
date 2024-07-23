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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = ops.sparse_segment_mean

    def construct(self, x, indices, segment_ids):
        return self.op(x, indices, segment_ids)


def sparse_segment_mean_numpy(x, indices, segment_ids):
    segment_num = int(max(segment_ids)) + 1
    output_shape = (segment_num,) + x.shape[1:]
    output = np.zeros(output_shape, dtype=x.dtype)
    for segment_id in range(segment_num):
        segment_indices = indices[np.equal(segment_ids, segment_id)]
        if segment_indices.shape[0] > 0:
            output[segment_id] = np.mean(x[segment_indices], axis=0)
    return output


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('data_type, index_type, error', [(np.float32, np.int32, 3), (np.float64, np.int64, 5)])
def test_net(data_type, index_type, error):
    """
    Feature: SparseSegmentMean operator.
    Description:  test cases for SparseSegmentMean operator.
    Expectation: the result match expectation.
    """
    data_shape = (4, 5, 6)
    index_shape = (5,)
    x = np.random.random(size=data_shape).astype(data_type)
    dim0 = x.shape[0]
    indices = np.random.randint(0, dim0, size=index_shape).astype(index_type)
    segment_ids = np.random.randint(0, 2 * dim0, size=index_shape).astype(index_type)
    segment_ids = np.array(sorted(segment_ids)).astype(index_type)
    np_out = sparse_segment_mean_numpy(x, indices, segment_ids)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    ms_out_py = Net()(Tensor(x), Tensor(indices), Tensor(segment_ids))
    np.testing.assert_almost_equal(ms_out_py.asnumpy(), np_out, decimal=error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms_out_graph = Net()(Tensor(x), Tensor(indices), Tensor(segment_ids))
    np.testing.assert_almost_equal(ms_out_graph.asnumpy(), np_out, decimal=error)

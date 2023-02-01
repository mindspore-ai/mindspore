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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.sparse_ops as P
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = P.SparseSegmentMeanWithNumSegments()

    def construct(self, x, indices, segment_ids, num_segments):
        return self.op(x, indices, segment_ids, num_segments)


def sparse_segment_mean_with_num_segments_numpy(x, indices, segment_ids, num_segments):
    output_shape = (num_segments,) + x.shape[1:]
    output = np.zeros(output_shape, dtype=x.dtype)
    for segment_id in range(num_segments):
        segment_indices = indices[np.equal(segment_ids, segment_id)]
        if segment_indices.shape[0] > 0:
            output[segment_id] = np.mean(x[segment_indices], axis=0)
    return output


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type, index_type, error', [(np.float16, np.int32, 1e-3),
                                                          (np.float32, np.int32, 1e-4), (np.float64, np.int64, 1e-5)])
def test_net(data_type, index_type, error):
    """
    Feature: SparseSegmentMeanWithNumSegments operator.
    Description:  test cases for SparseSegmentMeanWithNumSegments operator.
    Expectation: the result match expectation.
    """
    data_shape = (4, 5, 6)
    index_shape = (5,)
    num_segments = np.array(8).astype(index_type)
    x = np.random.random(size=data_shape).astype(data_type)
    indices = np.random.randint(
        0, data_shape[0], size=index_shape).astype(index_type)
    segment_ids = np.random.randint(
        0, num_segments, size=index_shape).astype(index_type)
    segment_ids = sorted(segment_ids)
    np_out = sparse_segment_mean_with_num_segments_numpy(
        x, indices, segment_ids, num_segments)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    ms_out_py = Net()(Tensor(x), Tensor(indices),
                      Tensor(segment_ids, dtype=Tensor(indices).dtype), Tensor(num_segments))
    np.testing.assert_allclose(ms_out_py.asnumpy(), np_out, error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms_out_graph = Net()(Tensor(x), Tensor(indices),
                         Tensor(segment_ids, dtype=Tensor(indices).dtype), Tensor(num_segments))
    np.testing.assert_allclose(
        ms_out_graph.asnumpy(), np_out, error)

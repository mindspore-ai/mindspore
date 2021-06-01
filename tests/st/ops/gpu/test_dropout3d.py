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
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import operations as P

class Dropout3DNet(nn.Cell):
    def __init__(self, keep_prob):
        super(Dropout3DNet, self).__init__()
        self.drop = P.Dropout3D(keep_prob)

    def construct(self, x):
        return self.drop(x)


def dropout_3d(keep_prob, nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_shape = [32, 16, 2, 5, 4]
    x_np = np.ones(x_shape).astype(nptype)
    dropout3d_net = Dropout3DNet(keep_prob)
    tx = Tensor(x_np)
    output, mask = dropout3d_net(tx)

    ## check output ##
    output_np = output.asnumpy()
    elem_count = x_np.size
    nonzero_count = np.count_nonzero(output_np)
    # assert correct proportion of elements kept
    assert (elem_count * (keep_prob - 0.1)) < nonzero_count < (elem_count * (keep_prob + 0.1))
    output_sum = np.sum(output_np)
    x_sum = np.sum(x_np)
    if keep_prob != 0.0:
        # assert output scaled correctly (expected value maintained)
        assert abs(output_sum - x_sum)/x_sum < 0.1

    ## check mask ##
    mask_np = mask.asnumpy()
    # specific to input with no zeros. Check for same number of nonzero elements
    assert np.count_nonzero(mask_np) == nonzero_count
    # check each channel is entirely True or False
    non_eq_chan = 0
    for n in range(mask_np.shape[0]):
        for c in range(mask_np.shape[1]):
            if not np.all(mask_np[n][c] == mask_np[n][c][0]):
                non_eq_chan = non_eq_chan + 1
    assert non_eq_chan == 0

    # check input, output, mask all have same shape
    assert x_np.shape == output_np.shape == mask_np.shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dropout3d_float16():
    dropout_3d(0.0, np.float16)
    dropout_3d(1.0, np.float16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dropout3d_float32():
    dropout_3d(0.0, np.float32)
    dropout_3d(1.0, np.float32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dropout3d_int8():
    dropout_3d(0.0, np.int8)
    dropout_3d(1.0, np.int8)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dropout3d_int16():
    dropout_3d(0.0, np.int16)
    dropout_3d(1.0, np.int16)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dropout3d_int32():
    dropout_3d(0.0, np.int32)
    dropout_3d(1.0, np.int32)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dropout3d_int64():
    dropout_3d(0.0, np.int64)
    dropout_3d(1.0, np.int64)

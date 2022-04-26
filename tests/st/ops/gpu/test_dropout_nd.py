# Copyright 2021-2022 Huawei Technologies Co., Ltd
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


def check_dropout_nd_by_keep_prob(input_x, output, output_mask, keep_prob):
    """
    Feature: check mindspore Dropout2D or Dropout3D's output and mask.
    Description: output shape, mask shap and keep_pro will be checked.
    Expectation: match to mindspore Dropout2D or Dropout3D.
    """
    # Check input, output, mask all have same shape
    assert input_x.shape == output.shape == output_mask.shape
    data_type = input_x.dtype
    error = 1e-6
    if data_type == np.float16:
        error = 1e-3
    data_shape = input_x.shape
    channels = data_shape[0] * data_shape[1]
    features = 1
    if len(input_x.shape) == 4:
        # HW
        features = features * data_shape[-2] * data_shape[-1]
    else:
        # DHW
        features = features * data_shape[-3] * data_shape[-2] * data_shape[-1]
    if keep_prob == 0.0:
        input_x_by_keep_prob = input_x.astype(data_type).reshape(channels, features)
    else:
        input_x_by_keep_prob = (input_x / keep_prob).astype(data_type).reshape(channels, features)
    output_reshape = output.reshape(channels, features)
    mask_reshape = output_mask.reshape(channels, features)
    # Check each channel is entirely True or False and output match to input_x
    for channel in range(channels):
        if np.all(output_reshape[channel] == 0):
            assert int(np.all(mask_reshape[channel])) == 0
        else:
            assert np.all(mask_reshape[channel])
            np.allclose(input_x_by_keep_prob[channel], output_reshape[channel], error, error)


class Dropout3DNet(nn.Cell):
    def __init__(self, keep_prob):
        super(Dropout3DNet, self).__init__()
        self.drop3d = P.Dropout3D(keep_prob)

    def construct(self, x):
        return self.drop3d(x)


class Dropout2DNet(nn.Cell):
    def __init__(self, keep_prob):
        super(Dropout2DNet, self).__init__()
        self.drop2d = P.Dropout2D(keep_prob)

    def construct(self, x):
        return self.drop2d(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("keep_prob", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("data_shape", [(32, 16, 4, 5), (32, 16, 2, 5, 4)])
@pytest.mark.parametrize("data_type", [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64])
def test_dropout_nd(data_shape, data_type, keep_prob):
    """
    Feature: Test Dropout2D and Dropout3D.
    Description: The input shape is 4d or 5d.
    Expectation: check it by function check_dropout_nd_by_keep_prob.
    """
    input_data = np.ones(data_shape).astype(data_type)
    if len(input_data.shape) == 4:
        dropout_nd = Dropout2DNet(keep_prob)
    else:
        dropout_nd = Dropout3DNet(keep_prob)
    output, mask = dropout_nd(Tensor(input_data))
    context.set_context(mode=context.GRAPH_MODE)
    check_dropout_nd_by_keep_prob(input_data, output.asnumpy(), mask.asnumpy(), keep_prob)
    context.set_context(mode=context.PYNATIVE_MODE)
    output, mask = dropout_nd(Tensor(input_data))
    check_dropout_nd_by_keep_prob(input_data, output.asnumpy(), mask.asnumpy(), keep_prob)

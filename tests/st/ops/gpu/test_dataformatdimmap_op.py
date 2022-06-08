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
from mindspore import Tensor
from mindspore.ops import operations as P


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_data_formata_dim_map_gpu():
    """
    Feature: DataFormatDimMapNet gpu kernel.
    Description: test the rightness of DataFormatDimMapNet gpu kernel.
    Expectation: Success.
    """
    x_np_1_gpu = np.array([-4, -3, -2, -1, 0, 1, 2, 3]).astype(np.int32)
    output_1_gpu = P.DataFormatDimMap()(Tensor(x_np_1_gpu))
    output_1_expect_gpu = np.array([0, 2, 3, 1, 0, 2, 3, 1]).astype(np.int32)
    assert np.allclose(output_1_gpu.asnumpy(), output_1_expect_gpu)

    output_2_gpu = P.DataFormatDimMap(src_format="NHWC", dst_format="NHWC")(Tensor(x_np_1_gpu))
    output_2_expect_gpu = np.array([0, 1, 2, 3, 0, 1, 2, 3]).astype(np.int32)
    assert np.allclose(output_2_gpu.asnumpy(), output_2_expect_gpu)

    output_3_gpu = P.DataFormatDimMap(src_format="NCHW", dst_format="NHWC")(Tensor(x_np_1_gpu))
    output_3_expect_gpu = np.array([0, 3, 1, 2, 0, 3, 1, 2]).astype(np.int32)
    assert np.allclose(output_3_gpu.asnumpy(), output_3_expect_gpu)

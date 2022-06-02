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
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_data_formata_dim_map():
    """
    Feature: DataFormatDimMapNet cpu kernel.
    Description: test the rightness of DataFormatDimMapNet cpu kernel.
    Expectation: Success.
    """
    x_np_1 = np.array([-4, -3, -2, -1, 0, 1, 2, 3]).astype(np.int32)
    output_1 = P.DataFormatDimMap()(Tensor(x_np_1))
    output_1_expect = np.array([0, 2, 3, 1, 0, 2, 3, 1]).astype(np.int32)
    assert np.allclose(output_1.asnumpy(), output_1_expect)

    output_2 = P.DataFormatDimMap(src_format="NHWC", dst_format="NHWC")(Tensor(x_np_1))
    output_2_expect = np.array([0, 1, 2, 3, 0, 1, 2, 3]).astype(np.int32)
    assert np.allclose(output_2.asnumpy(), output_2_expect)

    output_3 = P.DataFormatDimMap(src_format="NCHW", dst_format="NHWC")(Tensor(x_np_1))
    output_3_expect = np.array([0, 3, 1, 2, 0, 3, 1, 2]).astype(np.int32)
    assert np.allclose(output_3.asnumpy(), output_3_expect)

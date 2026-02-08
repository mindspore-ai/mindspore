# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Test set format interface for runtime.
"""
import numpy as np
from mindspore import Tensor, ops
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_assign_value():
    """
    Feature: Test format for tensor.
    Description: Test the passing of format in assign_value.
    Expectation: Run success.
    """
    np1 = np.random.randn(2, 2, 2, 2).astype(np.float32)
    x = Tensor(np1)
    y = ops.auto_generate.format_cast(x, 1)
    z = Tensor(np1)
    z.assign_value(y)
    assert np.allclose(z.asnumpy(), np1)

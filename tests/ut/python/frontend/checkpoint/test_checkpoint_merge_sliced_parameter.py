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
# ============================================================================
"""Test checkpoint."""
import numpy as np
from mindspore import Parameter, Tensor
from mindspore.train.serialization import merge_sliced_parameter


def test_checkpoint_merge_sliced_parameter_with_none_strategy():
    """
    Feature: test merge sliced parameter.
    Description: test merge sliced parameter.
    Expectation: the result match with expected result.
    """
    param_list = []
    np_list = []
    np_data = np.full((128, 96), 0.5, dtype=np.float32)
    param = Parameter(Tensor(np_data), name="test_param")
    for _ in range(8):
        param_list.append(param)
        np_list.append(np_data)
    out_ms = merge_sliced_parameter(param_list, strategy=None)
    out_np = np.concatenate(tuple(np_list), axis=0)
    np.allclose(out_np, out_ms.asnumpy(), 0, 0)

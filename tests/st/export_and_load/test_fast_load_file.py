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

import numpy as np
from safetensors.numpy import save_file, load_file
from mindspore.parallel.transform_safetensors import _fast_load_file
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1.00
          )
def test_safetensors_loading():
    """
    Feature: safetensors fast load.
    Description: test load safetensors.
    Expectation: Success.
    """
    array1 = np.random.randn(3, 3)
    array2 = np.random.randn(2, 2)

    data = {
        "array1": array1,
        "array2": array2
    }

    file_path = "example_numpy.safetensors"
    save_file(data, file_path)

    loaded_data = load_file(file_path)
    loaded_data_fast = _fast_load_file(file_path)
    assert set(loaded_data.keys()) == set(loaded_data_fast.keys())
    for key in loaded_data.keys():
        assert np.allclose(loaded_data[key], loaded_data_fast[key])

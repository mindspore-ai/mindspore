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

"""test from_numpy with special format"""

import numpy as np
import mindspore as ms
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_numpy_special_format():
    """
    Feature: test from numpy for special format
    Description: test special format of numpy to a tensor
    Expectation: success
    """
    # Define test data
    data = [1, 2, 3, 4, 5]
    float_data = [1.1, 2.2, 3.3]

    # Define dtypes with different byte orders
    # '<' : little-endian
    # '>' : big-endian
    # '=' : native endian (machine default)
    # '|' : not applicable (e.g., for boolean, string)
    # Note: '!' is an alias for '>' (big-endian)
    dtypes_to_test = [
        # Integer types
        ('int8', ['<i1', '>i1', '=i1', '|i1']),
        ('int16', ['<i2', '>i2', '=i2']),
        ('int32', ['<i4', '>i4', '=i4']),
        ('int64', ['<i8', '>i8', '=i8']),
        # Unsigned Integer types
        ('uint8', ['<u1', '>u1', '=u1', '|u1']),
        ('uint16', ['<u2', '>u2', '=u2']),
        ('uint32', ['<u4', '>u4', '=u4']),
        ('uint64', ['<u8', '>u8', '=u8']),
        # Float types
        ('float16', ['<f2', '>f2', '=f2']),
        ('float32', ['<f4', '>f4', '=f4']),
        ('float64', ['<f8', '>f8', '=f8']),
        # Boolean type (typically not affected by byte order)
        ('bool', ['<b1', '>b1', '=b1', '|b1']),
    ]

    for base_type, dtype_list in dtypes_to_test:
        for dtype_str in dtype_list:
            # Select appropriate data for the type
            if base_type.startswith('float'):
                test_data = float_data
            elif base_type.startswith('bool'):
                test_data = [True, False, True]
            else: # integer types
                test_data = data

            # Create numpy array with specific dtype/byte order
            np_arr = np.array(test_data, dtype=dtype_str)

            # Attempt to create MindSpore Tensor
            ms_tensor = ms.from_numpy(np_arr)

            assert np.allclose(ms_tensor.asnumpy(), np_arr)

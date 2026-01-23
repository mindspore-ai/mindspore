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
import mindspore as ms
from tests.mark_utils import arg_mark
from mindspore.common import dtype as mstype


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
def test_dtype_float32_another_float():
    """
    Feature: test mstype.float
    Description:
        1. Confirming that mstype.float and ms.float are equivalent to mstype.float32.
        2. Validating that the .to() method correctly converts a tensor to the
            specified dtype (mstype.float → float32).
        3. Checking the difference in dtype interpretation between string-based and
            enum-based arguments in the .astype() method:
        4. Using a string "float" results in float64.
        5. Using the mstype.float results in float32.
    Expectation: success.
    """
    a = ms.Tensor(1, dtype=mstype.float)
    b = ms.Tensor(2, dtype=ms.float)
    c = ms.Tensor(2, dtype=mstype.float16)
    d = c.to(mstype.float)
    e = c.astype("float")
    f = c.astype(mstype.float)
    assert a.dtype == mstype.float32
    assert b.dtype == mstype.float32
    assert d.dtype == mstype.float32
    assert e.dtype == mstype.float64
    assert f.dtype == mstype.float32

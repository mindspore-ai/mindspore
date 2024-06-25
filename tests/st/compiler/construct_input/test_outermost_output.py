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
""" test outermost outputs"""

import mindspore as ms


def test_none_in_outputs():
    """
    Feature: Return outputs with None.
    Description: The outermost network output has None.
    Expectation: No exception.
    """
    @ms.jit
    def func(x, y):
        return None, x + y, None

    x = ms.Tensor(1)
    y = ms.Tensor(2)
    out = func(x, y)
    assert out[0].asnumpy() == ms.Tensor(3).asnumpy()

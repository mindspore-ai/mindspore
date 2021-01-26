# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P

add = P.Add()
reshape = P.Reshape()


def test_shape_add(x1, x2, y1, y2, z1, z2):
    sum1 = add(x1, x2)
    sum2 = add(y1, y2)
    sum3 = add(z1, z2)
    reshape_sum1 = reshape(sum1, (2, 2, 3, 1))
    reshape_sum2 = reshape(sum2, (2, 2, 3, 1))
    reshape_sum3 = reshape(sum3, (2, 2, 3, 1))
    result = add(reshape_sum1, reshape_sum2)
    result = add(result, reshape_sum3)
    return result

# Copyright 2023 Huawei Technologies Co., Ltd
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
import os
import mindspore as ms


def test_constexpr_validation():
    """
    Feature: Optimization of constexpr
    Description: If use constexpr decorate for loop, the compile performance will be improved. If constexpr is not
        validate, 'exceed limit 1000' exception wil be raised.
    Expectation: No 'exceed limit 1000' exception raised.
    """

    @ms.constexpr
    def for_loop_calculate(range_num):
        out = 0
        for i in range(range_num):
            if i % 2 == 0 and i % 7 != 0:
                out = out + i
        return out // range_num

    @ms.jit
    def func(x):
        new_shape = for_loop_calculate(100000)
        return ms.ops.broadcast_to(x, (new_shape,))

    reserved_env = os.getenv('MS_DEV_PRECOMPILE_ONLY')
    os.environ['MS_DEV_PRECOMPILE_ONLY'] = '1'
    out = func(ms.Tensor([1]))
    print("out:", out)
    if reserved_env is None:
        os.unsetenv('MS_DEV_PRECOMPILE_ONLY')
    else:
        os.environ['MS_DEV_PRECOMPILE_ONLY'] = reserved_env

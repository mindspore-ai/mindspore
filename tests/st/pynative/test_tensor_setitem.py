# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test_tensor_setitem """
import numpy as onp
import pytest

from mindspore import Tensor, context
from mindspore.nn import Cell


def setup_module():
    context.set_context(mode=context.GRAPH_MODE)


def setup_testcase(input_np, case_fn):
    input_ms = Tensor(input_np)

    class TensorSetItem(Cell):
        def construct(self, x):
            return case_fn(x)

    class NumpySetItem():
        def __call__(self, x):
            return case_fn(x)

    out_ms = TensorSetItem()(input_ms)
    out_np = NumpySetItem()(input_np)
    assert onp.all(out_ms.asnumpy() == out_np)


class TensorSetItemByList(Cell):
    def construct(self, x):
        x[[0, 1], [1, 2], [1, 3]] = [3, 4]
        x[([0, 1], [0, 2], [1, 1])] = [10, 5]
        x[[0, 1], ..., [0, 1]] = 4
        return x

class NumpySetItemByList():
    def __call__(self, x):
        x[[0, 1], [1, 2], [1, 3]] = [3, 4]
        x[([0, 1], [0, 2], [1, 1])] = [10, 5]
        x[[0, 1], ..., [0, 1]] = 4
        return x

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_list():
    x = onp.ones((2, 3, 4), dtype=onp.float32)
    def cases(x):
        x[[0, 1], [1, 2], [1, 3]] = [3, 4]
        x[([0, 1], [0, 2], [1, 1])] = [10, 5]
        x[[0, 1], ..., [0, 1]] = 4
        return x
    setup_testcase(x, cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_with_sequence():
    x = onp.ones((2, 3, 4), dtype=onp.float32)
    def cases(x):
        x[...] = [3]
        x[..., 1] = ([1, 2, 3], [4, 5, 6])
        x[0] = ((0, 1, 2, 3), (4, 5, 6, 7), [8, 9, 10, 11])
        x[1:2] = ((0, 1, 2, 3), (4, 5, 6, 7), [8, 9, 10, 11])
        return x
    setup_testcase(x, cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_dtype():
    x = onp.ones((2, 3, 4), dtype=onp.float32)
    def cases(x):
        x[...] = 3
        x[..., 1] = 3.0
        x[0] = True
        x[1:2] = ((0, False, 2, 3), (4.0, 5, 6, 7), [True, 9, 10, 11])
        return x
    setup_testcase(x, cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_tuple_with_int():
    x = onp.arange(24).reshape(2, 3, 4).astype(onp.float32)
    def cases(x):
        x[..., 2, False, 1] = -1
        x[0, True, 0, None, True] = -2
        x[0, ..., None] = -3
        x[..., 0, None, 1, True, True, None] = -4
        return x
    setup_testcase(x, cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_tuple_with_list():
    x = onp.arange(24).reshape(2, 3, 4).astype(onp.float32)
    def cases(x):
        x[..., 2, False, 1] = [-1]
        x[0, True, 0, None, True] = [-2, -2, -2, -2]
        x[0, ..., None] = [[-3], [-3], [-3], [-3]]
        x[..., 0, None, 1, True, True, None] = [[[-4]], [[-4]]]
        return x
    setup_testcase(x, cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_nested_unit_list():
    x = onp.arange(24).reshape(2, 3, 4).astype(onp.float32)
    def cases(x):
        x[[[[0]]], True] = -1
        x[[1], ..., [[[[2]]]]] = -2
        x[0, [[[2]]], [1]] = -3
        return x
    setup_testcase(x, cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_with_broadcast():
    x = onp.arange(2*3*4*5*6).reshape(2, 3, 4, 5, 6).astype(onp.float32)
    v1 = onp.full((1, 4, 5), -1).tolist()
    v2 = onp.full((4, 1, 6), -2).tolist()
    def cases(x):
        x[..., 4] = v1
        x[0, 2] = v2
        x[1, 0, ..., 3] = [[-3], [-3], [-3], [-3]]
        x[0, ..., 1, 3, 5] = -4
        return x
    setup_testcase(x, cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_mul_by_scalar():
    x = onp.ones((4, 5), dtype=onp.float32)
    def cases(x):
        x[1, :] = x[1, :]*2
        x[:, 2] = x[:, 3]*3.0
        return x
    setup_testcase(x, cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_slice():
    x = onp.ones((3, 4, 5), dtype=onp.float32)
    def cases(x):
        x[1:2] = 2
        x[-3:1] = 3
        x[-10:3:2] = 4
        x[5:0:3] = 5
        x[5:5:5] = 6
        x[-1:2] = 7
        return x
    setup_testcase(x, cases)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_tuple_of_slices():
    x = onp.ones((3, 4, 5), dtype=onp.float32)
    def cases(x):
        x[1:2, 2] = 2
        x[0, -4:1] = 3
        x[1, -10:3:2] = 4
        x[5:0:3, 3] = 5
        x[1:1, 2:2] = 6
        return x
    setup_testcase(x, cases)

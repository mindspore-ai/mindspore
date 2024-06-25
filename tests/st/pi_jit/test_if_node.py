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
"""test Control flow(if) implement"""
import sys
import pytest
import mindspore.context as context
from mindspore import Tensor, jit
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark

SYS_VER = (sys.version_info.major, sys.version_info.minor)
if SYS_VER != (3, 7) and SYS_VER != (3, 9):
    pytest.skip(reason="not implement for python" + str(SYS_VER), allow_module_level=True)


@jit(mode="PIJit", jit_config={"compile_without_capture": True})
def single_branch(x, y):
    if x > 0:
        x = x + y

    return x

@jit(mode="PIJit", jit_config={"compile_without_capture": True})
def repeat_single_branch(x, y):
    if x > 0:
        x = x + y

    x = x + x

    if y is not None:
        y = 2 * y

    x = x + y

    return x

@jit(mode="PIJit", jit_config={"compile_without_capture": True})
def nest_single_branch(x, y):
    if x > 0:
        x = x + y
        if y > 5:
            y = 3 * y
        x = x + y

    return x

@jit(mode="PIJit", jit_config={"compile_without_capture": True})
def full_branch(x, y):
    if x > 0:
        x = x + y
    else:
        x = x - y

    return x

@jit(mode="PIJit", jit_config={"compile_without_capture": True})
def repeat_full_branch(x, y):
    if x > 0:
        x = x + y
    else:
        x = x - y

    if x > 0:
        x = x + y
    else:
        x = x - y

    return x

@jit(mode="PIJit", jit_config={"compile_without_capture": True})
def nest_full_branch(x, y):
    if x > 0:
        x = x + y
        if x > 0:
            x = x * 2
        else:
            x = x * 3
    else:
        x = x - y
        if x > 0:
            x = x * 5
        else:
            x = x * 6

    return x

@jit(mode="PIJit", jit_config={"compile_without_capture": True})
def multi_branch(x, y):
    if x > 0:
        x = x + y
    elif x == 0:
        x = y
    else:
        x = x - y

    return x

@jit(mode="PIJit", jit_config={"compile_without_capture": True})
def return_branch_1(x, y):
    if x > 0:
        return x + y

    return x

# pylint: disable=R1705
@jit(mode="PIJit", jit_config={"compile_without_capture": True})
def return_branch_2(x, y):
    if x > 0:
        return x + y
    else:
        x = x - y

    return x

# pylint: disable=R1705
@jit(mode="PIJit", jit_config={"compile_without_capture": True})
def return_branch_3(x, y):
    if x > 0:
        return x + y
    else:
        return x - y

    return x

# pylint: disable=R1705
@jit(mode="PIJit", jit_config={"compile_without_capture": True})
def return_branch_4(x, y):
    if x > 0:
        return x + y
    else:
        return x

# pylint: disable=R1705
@jit(mode="PIJit", jit_config={"compile_without_capture": True})
def return_branch_5(x, y):
    if x > 0:
        if x > 5:
            return x + y
        else:
            return x * 5
    else:
        if x < -10:
            return x
        else:
            return x * 10


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('jit_func', [single_branch])
def test_single_branch(jit_func):
    """
    Feature: Control flow(if) implement
    Description: test single branch.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(6, mstype.float32)
    y = Tensor(10, mstype.float32)
    assert jit_func(x, y) == Tensor(16, mstype.float32)
    x = Tensor(0, mstype.float32)
    assert jit_func(x, y) == Tensor(0, mstype.float32)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('jit_func', [repeat_single_branch])
def test_repeat_single_branch(jit_func):
    """
    Feature: Control flow(if) implement
    Description: test single branch.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(6, mstype.float32)
    y = Tensor(10, mstype.float32)
    assert jit_func(x, y) == Tensor(52, mstype.float32)
    x = Tensor(0, mstype.float32)
    assert jit_func(x, y) == Tensor(20, mstype.float32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('jit_func', [nest_single_branch])
def test_nest_single_branch(jit_func):
    """
    Feature: Control flow(if) implement
    Description: test single branch.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(6, mstype.float32)
    y = Tensor(10, mstype.float32)
    assert jit_func(x, y) == Tensor(46, mstype.float32)
    x = Tensor(0, mstype.float32)
    assert jit_func(x, y) == Tensor(0, mstype.float32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('jit_func', [full_branch])
def test_full_branch(jit_func):
    """
    Feature: Control flow(if) implement
    Description: test full branch.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(6, mstype.float32)
    y = Tensor(10, mstype.float32)
    assert jit_func(x, y) == Tensor(16, mstype.float32)
    x = Tensor(0, mstype.float32)
    assert jit_func(x, y) == Tensor(-10, mstype.float32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('jit_func', [repeat_full_branch])
def test_repeat_full_branch(jit_func):
    """
    Feature: Control flow(if) implement
    Description: test full branch.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(6, mstype.float32)
    y = Tensor(10, mstype.float32)
    assert jit_func(x, y) == Tensor(26, mstype.float32)
    x = Tensor(0, mstype.float32)
    assert jit_func(x, y) == Tensor(-20, mstype.float32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('jit_func', [nest_full_branch])
def test_nest_full_branch(jit_func):
    """
    Feature: Control flow(if) implement
    Description: test full branch.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(6, mstype.float32)
    y = Tensor(10, mstype.float32)
    assert jit_func(x, y) == Tensor(32, mstype.float32)
    x = Tensor(0, mstype.float32)
    assert jit_func(x, y) == Tensor(-60, mstype.float32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('jit_func', [multi_branch])
def test_multi_branch(jit_func):
    """
    Feature: Control flow(if) implement
    Description: test multi branch.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(6, mstype.float32)
    y = Tensor(10, mstype.float32)
    assert jit_func(x, y) == Tensor(16, mstype.float32)
    x = Tensor(0, mstype.float32)
    assert jit_func(x, y) == Tensor(10, mstype.float32)
    x = Tensor(-6, mstype.float32)
    assert jit_func(x, y) == Tensor(-16, mstype.float32)

@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('jit_func', [return_branch_1])
def test_return_branch_1(jit_func):
    """
    Feature: Control flow(if) implement
    Description: test return branch.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(6, mstype.float32)
    y = Tensor(10, mstype.float32)
    assert jit_func(x, y) == Tensor(16, mstype.float32)
    x = Tensor(-6, mstype.float32)
    assert jit_func(x, y) == Tensor(-6, mstype.float32)

@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('jit_func', [return_branch_2])
def test_return_branch_2(jit_func):
    """
    Feature: Control flow(if) implement
    Description: test return branch.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(6, mstype.float32)
    y = Tensor(10, mstype.float32)
    assert jit_func(x, y) == Tensor(16, mstype.float32)
    x = Tensor(-6, mstype.float32)
    assert jit_func(x, y) == Tensor(-16, mstype.float32)

@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('jit_func', [return_branch_3])
def test_return_branch_3(jit_func):
    """
    Feature: Control flow(if) implement
    Description: test return branch.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(6, mstype.float32)
    y = Tensor(10, mstype.float32)
    assert jit_func(x, y) == Tensor(16, mstype.float32)
    x = Tensor(-6, mstype.float32)
    assert jit_func(x, y) == Tensor(-16, mstype.float32)

@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('jit_func', [return_branch_4])
def test_return_branch_4(jit_func):
    """
    Feature: Control flow(if) implement
    Description: test return branch.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(6, mstype.float32)
    y = Tensor(10, mstype.float32)
    assert jit_func(x, y) == Tensor(16, mstype.float32)
    x = Tensor(-6, mstype.float32)
    assert jit_func(x, y) == Tensor(-6, mstype.float32)

@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('jit_func', [return_branch_5])
def test_return_branch_5(jit_func):
    """
    Feature: Control flow(if) implement
    Description: test return branch.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(6, mstype.float32)
    y = Tensor(10, mstype.float32)
    assert jit_func(x, y) == Tensor(16, mstype.float32)
    x = Tensor(2, mstype.float32)
    assert jit_func(x, y) == Tensor(10, mstype.float32)
    x = Tensor(-16, mstype.float32)
    assert jit_func(x, y) == Tensor(-16, mstype.float32)
    x = Tensor(-6, mstype.float32)
    assert jit_func(x, y) == Tensor(-60, mstype.float32)

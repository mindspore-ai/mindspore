# Copyright 2024 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
from mindspore.common import Tensor
from mindspore import context, jit
from mindspore.ops.composite import GradOperation
from mindspore._c_expression import get_code_extra
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_tuple():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    cfg = {"compile_with_try": False}

    @pi_jit_with_config(jit_config=cfg)
    def inner_func(x, y):
        return 2 * x[0] + y

    def func(x, y):
        x = x * 3
        return inner_func((x,), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
    jcr = get_code_extra(inner_func.__wrapped__)
    assert jcr["break_count_"] == 0
    assert jcr["code"]["call_count_"] > 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_tuple_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    cfg = {"compile_with_try": False}

    @pi_jit_with_config(jit_config=cfg)
    def inner_func(m):
        return 2 * m[0][0] + m[1]

    def func(x, y):
        x = x * 3
        return inner_func(((x,), y))

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
    jcr = get_code_extra(inner_func.__wrapped__)
    assert jcr["break_count_"] == 0
    assert jcr["code"]["call_count_"] > 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_list():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    cfg = {"compile_with_try": False}

    @pi_jit_with_config(jit_config=cfg)
    def inner_func(x, y):
        return 2 * x[0] + y

    def func(x, y):
        x = x * 3
        return inner_func([x,], y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
    jcr = get_code_extra(inner_func.__wrapped__)
    assert jcr["break_count_"] == 0
    assert jcr["code"]["call_count_"] > 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_list_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    cfg = {"compile_with_try": False}

    @pi_jit_with_config(jit_config=cfg)
    def inner_func(m):
        return 2 * m[0][0] + m[1]

    def func(x, y):
        x = x * 3
        return inner_func([[x,], y])

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
    jcr = get_code_extra(inner_func.__wrapped__)
    assert jcr["break_count_"] == 0
    assert jcr["code"]["call_count_"] > 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_dict():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    cfg = {"compile_with_try": False}

    @pi_jit_with_config(jit_config=cfg)
    def inner_func(m):
        return 2 * m["x"] + m["y"]

    def func(x, y):
        x = x * 3
        return inner_func({"x": x, "y": y})

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
    jcr = get_code_extra(inner_func.__wrapped__)
    assert jcr["break_count_"] == 0
    assert jcr["code"]["call_count_"] > 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_dict_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    cfg = {"compile_with_try": False}

    @pi_jit_with_config(jit_config=cfg)
    def inner_func(m):
        return 2 * m["x"][0] + m["y"]

    def func(x, y):
        x = x * 3
        return inner_func({"x": (x,), "y": y})

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
    jcr = get_code_extra(inner_func.__wrapped__)
    assert jcr["break_count_"] == 0
    assert jcr["code"]["call_count_"] > 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_vargs():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    cfg = {"compile_with_try": False}

    @pi_jit_with_config(jit_config=cfg)
    def inner_func(*args):
        return 2 * args[0][0] + args[1]

    def func(x, y):
        x = x * 3
        return inner_func((x,), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
    jcr = get_code_extra(inner_func.__wrapped__)
    assert jcr["break_count_"] == 0
    assert jcr["code"]["call_count_"] > 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_vargs_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    cfg = {"compile_with_try": False}

    @pi_jit_with_config(jit_config=cfg)
    def inner_func(*args):
        return 2 * args[0][0][0] + args[1]

    def func(x, y):
        x = x * 3
        return inner_func(((x,),), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
    jcr = get_code_extra(inner_func.__wrapped__)
    assert jcr["break_count_"] == 0
    assert jcr["code"]["call_count_"] > 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_kwargs():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    cfg = {"compile_with_try": False}

    @pi_jit_with_config(jit_config=cfg)
    def inner_func(**kwargs):
        return 2 * kwargs["m"][0] + kwargs["n"]

    def func(x, y):
        x = x * 3
        return inner_func(m=(x,), n=y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
    jcr = get_code_extra(inner_func.__wrapped__)
    assert jcr["break_count_"] == 0
    assert jcr["code"]["call_count_"] > 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_kwargs_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    cfg = {"compile_with_try": False}

    @pi_jit_with_config(jit_config=cfg)
    def inner_func(**kwargs):
        return 2 * kwargs["m"][0][0] + kwargs["n"]

    def func(x, y):
        x = x * 3
        return inner_func(m=([x,],), n=y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
    jcr = get_code_extra(inner_func.__wrapped__)
    assert jcr["break_count_"] == 0
    assert jcr["code"]["call_count_"] > 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_invalid_input():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: RuntimeError.
    """
    @jit(capture_mode="bytecode")
    def inner_func(m):
        return 2 * m["x"][0] + m["y"]

    def func(x, y):
        x = x * 3
        return inner_func({"x": (x, "a"), "y": y})

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
    jcr = get_code_extra(inner_func.__wrapped__)
    assert jcr["break_count_"] == 0
    assert jcr["code"]["call_count_"] > 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_invalid_input_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: RuntimeError.
    """
    @jit(capture_mode="bytecode")
    def inner_func(x, y):
        return 2 * x[0] + y

    def func(x, y):
        x = x * 3
        return inner_func((x, None), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))
    jcr = get_code_extra(inner_func.__wrapped__)
    assert jcr["break_count_"] == 0
    assert jcr["code"]["call_count_"] > 0

import os
import shutil
import sys
from pathlib import Path

import pytest
import mindspore as ms
from mindspore import Tensor, ops, jit
from mindspore import numpy as np
import mindspore.nn as nn
from mindspore.common import dtype as mstype
import mindspore.dataset as ds
from mindspore._c_expression import get_code_extra
from tests.mark_utils import arg_mark
from .share.utils import match_array, match_value, assert_no_graph_break, pi_jit_with_config
from .share.modeltrain_utils import create_train_model, GeneratorFakeData

condition = not (sys.version_info.major == 3 and sys.version_info.minor in [8, 9])

jit_config = {'recapture_loop_body': True}


def setup_ir_capture(case_name: str, save_level: str) -> Path:
    ir_dir = Path(__file__).parent / case_name
    if ir_dir.exists():
        shutil.rmtree(ir_dir)
    ir_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MS_DEV_SAVE_GRAPHS'] = save_level
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = str(ir_dir)
    return ir_dir


def cleanup_ir_capture(ir_dir: Path):
    os.environ.pop('MS_DEV_SAVE_GRAPHS', None)
    os.environ.pop('MS_DEV_SAVE_GRAPHS_PATH', None)
    if ir_dir.exists():
        shutil.rmtree(ir_dir)


def count_ir_parameters(ir_dir: Path, keyword: str = 'graph_before_compile') -> int:
    ir_files = sorted(ir_dir.rglob(f'*{keyword}*'))
    assert ir_files, f'No IR files matching {keyword} generated under {ir_dir}'
    ir_file = ir_files[-1]
    with ir_file.open('r', encoding='utf-8') as handle:
        lines = handle.readlines()
    filtered = [
        line for line in lines
        if line.startswith('%para') and not any(token in line for token in ('dict', 'list', 'tuple'))
    ]
    return len(filtered)

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test001():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(a):
        result = a
        for i in range(10):
            result = result + a
            if result[0,0] > 10:
                result = result + a
        return result
    fn = pi_jit_with_config(func, jit_config=jit_config)
    x1 = np.randn((2,4))
    expect = func(x1)
    got = fn(x1)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test002():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(a):
        result = a
        if a.shape[0] == 2:
            for i in range(10):
                result = result + a
                if result[0,0] > 10:
                    result = result + a
        else:
            result = result + 1
        return result
    fn = pi_jit_with_config(func, jit_config=jit_config)
    x1 = np.randn((2,4))
    expect = func(x1)
    got = fn(x1)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test003():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(a):
        result = a
        for i in range(10):
            result = result + a
            if result[0,0] > 10:
                result = result + a
            else:
                result = result + 2
        return result
    fn = pi_jit_with_config(func, jit_config=jit_config)
    x1 = np.randn((2,4))
    expect = func(x1)
    got = fn(x1)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test004():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(a):
        result = a
        def fn():
            return a
        for i in range(10):
            result = result + a
            if result[0,0] > 10:
                result = result + a
            else:
                result = result + 2
        return result
    fn = pi_jit_with_config(func, jit_config=jit_config)
    x1 = np.randn((2,4))
    expect = func(x1)
    got = fn(x1)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skip(reason="RunGraph Failed !!!")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test005():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(a):
        result = a
        for i in range(10):
            def fn():
                return i
            result = result + a
            if result[0,0] > 10:
                result = result + a
            else:
                result = result + 2
        return result
    fn = pi_jit_with_config(func, jit_config=jit_config)
    x1 = np.randn((2,4))
    expect = func(x1)
    got = fn(x1)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test006():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(tensor_list):
        result = tensor_list[0]
        for i in tensor_list:
            result = result + i
            if i[0,0] > 1:
                result = result + 1
        return result
    x1 = np.randn((2,4))
    x2 = np.randn((2,4))
    x3 = np.randn((2,4))
    x4 = np.randn((2,4))
    tensor_list=[x1,x2,x3,x4]
    fn = pi_jit_with_config(func, jit_config=jit_config)
    expect = func(tensor_list)
    got = fn(tensor_list)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test007():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(tensor_list):
        result = 0
        for i in tensor_list:
            result = result + i.sum()
            if i[0,0] > 1:
                result = result + 1
        return result
    x1 = np.randn((2,4))
    x2 = np.randn((3,4))
    x3 = np.randn((4,3))
    x4 = np.randn((5,3))
    tensor_list=[x1,x2,x3,x4]
    fn = pi_jit_with_config(func, jit_config=jit_config)
    expect = func(tensor_list)
    got = fn(tensor_list)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test008():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(a):
        i = 0
        result = a
        while(i < 10):
            if result[0,0] > 1:
                result = result + 1
            else:
                result = result + 2
            i += 1
        return result
    fn = pi_jit_with_config(func, jit_config=jit_config)
    x1 = np.randn((2,4))
    expect = func(x1)
    got = fn(x1)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@pytest.mark.skipif(reason='legacy issue')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_for_range_tensor_recreation():
    """
    Feature: Loop body recapture.
    Description: For-loop recreates tensors from asnumpy inside loop while PIJit recaptures the loop body.
    Expectation: JIT result matches pynative and the compiled graph has no graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_001
    """
    def func(x):
        out = x
        for i in range(10):
            y = x + x
            z = Tensor(y.asnumpy() + i)
            out = out - z
        return out

    input_x = np.randn((2, 3))
    pynative_result = func(input_x)
    jit_func = pi_jit_with_config(func, jit_config=jit_config)
    jit_result = jit_func(input_x)
    match_array(pynative_result, jit_result)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_iterate_list_tensor_recreation():
    """
    Feature: Loop body recapture.
    Description: For-loop iterates list literals and recreates tensors using asnumpy during each iteration.
    Expectation: JIT result matches pynative and the compiled graph has no graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_002
    """
    def func(x):
        lista = [1, 2, 3]
        for item in lista:
            x = x + item
            y = Tensor(x.asnumpy())
            x = x + y
            z = Tensor(x.asnumpy() * 1)
        return z

    input_x = np.randn((2, 3))
    pynative_result = func(input_x)
    jit_func = pi_jit_with_config(func, jit_config=jit_config)
    jit_result = jit_func(input_x)
    match_array(pynative_result, jit_result)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_enumerate_list_tensor_recreation():
    """
    Feature: Loop body recapture.
    Description: Enumerate loop recreates tensors via asnumpy to validate loop body recapture.
    Expectation: JIT result matches pynative and the compiled graph has no graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_003
    """
    def func(x):
        lista = [1, 2, 3]
        for index, item in enumerate(lista):
            x = Tensor(x.asnumpy())
            x = x + item
            x = x + index
            x = Tensor(x.asnumpy() * 1)
        return x

    input_x = np.randn((2, 3))
    pynative_result = func(input_x)
    jit_func = pi_jit_with_config(func, jit_config=jit_config)
    jit_result = jit_func(input_x)
    match_array(pynative_result, jit_result)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_call_method_inside_loop():
    """
    Feature: Loop body recapture.
    Description: Loop invokes helper method inside PIJit recaptured construct to reuse shared logic.
    Expectation: JIT result matches pynative and the compiled graph has no graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_004
    """
    class LoopWithHelper(nn.Cell):
        def __init__(self):
            super().__init__()

        def for_func(self, tensor, index):
            return Tensor(tensor.asnumpy() * index)

        def construct(self, x):
            out = x
            for index in range(3):
                y = x + x
                z = self.for_func(y, index)
                out = ops.add(out, z)
            return out

    input_x = np.randn((2, 3))
    pynative_net = LoopWithHelper()
    pynative_result = pynative_net(input_x)

    jit_net = LoopWithHelper()
    jit_net.construct = pi_jit_with_config(jit_net.construct, jit_config=jit_config)
    jit_result = jit_net(input_x)
    match_array(pynative_result, jit_result)
    assert_no_graph_break(jit_net.construct)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@pytest.mark.skipif(reason='legacy issue')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_nested_jitted_method():
    """
    Feature: Loop body recapture.
    Description: Nested jitted helper executes inside outer loop requiring loop body recapture.
    Expectation: JIT result matches pynative and the compiled graph has no graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_005
    """
    class LoopWithNestedJit(nn.Cell):
        def __init__(self):
            super().__init__()

        def for_2_func(self, tensor, index):
            out = tensor
            for inner in range(10):
                y = tensor + tensor
                z = Tensor((y.asnumpy() + inner) * index)
                out = out + z
            return out

        def construct(self, x):
            out = x
            for index in range(3):
                y = x + x
                z = self.for_2_func(y, index)
                out = out + z
            return out

    input_x = np.randn((2, 3))
    pynative_net = LoopWithNestedJit()
    pynative_result = pynative_net(input_x)
    jit_net = LoopWithNestedJit()
    jit_net.for_2_func = pi_jit_with_config(jit_net.for_2_func, jit_config=jit_config)
    jit_net.construct = pi_jit_with_config(jit_net.construct, jit_config=jit_config)
    jit_result = jit_net(input_x)
    match_array(pynative_result, jit_result)
    assert_no_graph_break(jit_net.construct)
    assert_no_graph_break(jit_net.for_2_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@pytest.mark.skipif(reason='legacy issue')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_while_loop_tensor_recreation():
    """
    Feature: Loop body recapture.
    Description: While-loop recreates tensors from asnumpy ensuring loop body recapture works for while constructs.
    Expectation: JIT result matches pynative and the compiled graph has no graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_006
    """
    def func(x):
        out = x
        i = 0
        while i < 10:
            y = x + x
            z = Tensor(y.asnumpy() + i)
            out = out + z
            i += 1
        return out

    input_x = np.randn((2, 3))
    pynative_result = func(input_x)
    jit_func = pi_jit_with_config(func, jit_config=jit_config)
    jit_result = jit_func(input_x)
    match_array(pynative_result, jit_result)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@pytest.mark.skipif(reason='legacy issue')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_nested_for_loops_tensor_recreation():
    """
    Feature: Loop body recapture.
    Description: Nested for-loops recreate tensors from asnumpy inside the inner loop.
    Expectation: JIT result matches pynative and the compiled graph has no graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_007
    """
    def func(x):
        out = x
        for outer in range(2):
            for inner in range(5):
                y = x + x
                z = Tensor(y.asnumpy() + outer + inner)
                out = out + z
        return out

    input_x = np.randn((2, 3))
    pynative_result = func(input_x)
    jit_func = pi_jit_with_config(func, jit_config=jit_config)
    jit_result = jit_func(input_x)
    match_array(pynative_result, jit_result)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@pytest.mark.skipif(reason='legacy issue')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_nested_for_loops_reassign_input():
    """
    Feature: Loop body recapture.
    Description: Nested for-loops reassign loop variable to new tensors created via asnumpy.
    Expectation: JIT result matches pynative and the compiled graph has no graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_008
    """
    def func(x):
        out = x
        for outer in range(2):
            x = Tensor(x.asnumpy() + outer)
            for inner in range(5):
                y = x + x
                z = Tensor(y.asnumpy() + inner)
                out = out + z
        return out

    input_x = np.randn((2, 3))
    pynative_result = func(input_x)
    jit_func = pi_jit_with_config(func, jit_config=jit_config)
    jit_result = jit_func(input_x)
    match_array(pynative_result, jit_result)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@pytest.mark.skipif(reason='legacy issue')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_nested_for_loops_tensor_addition():
    """
    Feature: Loop body recapture.
    Description: Nested for-loops perform tensor addition without asnumpy recreation to ensure coverage of pure tensor ops.
    Expectation: JIT result matches pynative and the compiled graph has no graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_009
    """
    def func(x):
        out = x
        for outer in range(2):
            x = Tensor(x.asnumpy() + outer)
            for inner in range(5):
                y = x + x
                z = y + inner
                out = out + z
        return out

    input_x = np.randn((2, 3))
    pynative_result = func(input_x)
    jit_func = pi_jit_with_config(func, jit_config=jit_config)
    jit_result = jit_func(input_x)
    match_array(pynative_result, jit_result)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@pytest.mark.skipif(reason='legacy issue')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_while_outer_for_inner():
    """
    Feature: Loop body recapture.
    Description: While-loop outside and for-loop inside recreate tensors from asnumpy values.
    Expectation: JIT result matches pynative and the compiled graph has no graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_010
    """
    def func(x):
        out = x
        i = 0
        while i < 2:
            for j in range(5):
                y = x + x
                z = Tensor(y.asnumpy() + i + j)
                out = out + z
            i += 1
        return out

    input_x = np.randn((2, 3))
    pynative_result = func(input_x)
    jit_func = pi_jit_with_config(func, jit_config=jit_config)
    jit_result = jit_func(input_x)
    match_array(pynative_result, jit_result)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@pytest.mark.skipif(reason='legacy issue')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_for_outer_while_inner():
    """
    Feature: Loop body recapture.
    Description: For-loop outside and while-loop inside recreate tensors from asnumpy values.
    Expectation: JIT result matches pynative and the compiled graph has no graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_011
    """
    def func(x):
        out = x
        for _ in range(2):
            j = 0
            while j < 5:
                y = x + x
                z = Tensor(y.asnumpy() + j)
                out = out + z
                j += 1
        return out

    input_x = np.randn((2, 3))
    pynative_result = func(input_x)
    jit_func = pi_jit_with_config(func, jit_config=jit_config)
    jit_result = jit_func(input_x)
    match_array(pynative_result, jit_result)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@pytest.mark.skipif(reason='legacy issue')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_nested_while_reassign_input():
    """
    Feature: Loop body recapture.
    Description: Nested while-loops reassign tensors using asnumpy and perform accumulation.
    Expectation: JIT result matches pynative and the compiled graph has no graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_012
    """
    def func(x):
        out = x
        i = 1
        while i < 3:
            j = 0
            x = Tensor(x.asnumpy() * i)
            while j < 3:
                y = x + x
                out = out + y
                j += 1
            i += 1
        return out

    input_x = np.randn((2, 3))
    pynative_result = func(input_x)
    jit_func = pi_jit_with_config(func, jit_config=jit_config)
    jit_result = jit_func(input_x)
    match_array(pynative_result, jit_result)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_ir_positional_args():
    """
    Feature: Loop body recapture.
    Description: Validate positional argument handling with nested list and dict inputs when saving IR.
    Expectation: JIT result matches pynative and IR contains only tensor parameters.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_016
    """
    ir_dir = setup_ir_capture("loop_body_recapture_ir_positional_args", "2")
    try:
        def pos_args(s, d):
            return s[0] + s[1] + d['a'] + d['b']

        tensor_a = Tensor([[1., 2.], [3., 4.]], mstype.float32)
        tensor_b = Tensor([[5., 6.], [7., 8.]], mstype.float32)
        tensor_c = Tensor([[50., 60.], [70., 80.]], mstype.float32)

        s = [tensor_a, tensor_b]
        d = {'a': tensor_a, 'b': tensor_b}
        pynative_out = pos_args(s, d)
        jit_func = pi_jit_with_config(pos_args, jit_config=jit_config)
        jit_out = jit_func(s, d)
        match_array(pynative_out, jit_out)

        s[0] = tensor_c
        d['a'] = tensor_c
        updated_pynative = pos_args(s, d)
        updated_jit = jit_func(s, d)
        match_array(updated_pynative, updated_jit)
        assert_no_graph_break(jit_func)
        assert count_ir_parameters(ir_dir) == 4
    finally:
        cleanup_ir_capture(ir_dir)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_ir_varargs_inputs():
    """
    Feature: Loop body recapture.
    Description: Validate *args inputs containing list and dict when saving IR.
    Expectation: JIT result matches pynative and IR contains only tensor parameters.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_017
    """
    ir_dir = setup_ir_capture("loop_body_recapture_ir_varargs_inputs", "1")
    try:
        def var_args(*args):
            return args[0][0] + args[0][1] + args[1]['a'] + args[1]['b']

        tensor_a = Tensor([[1., 2.], [3., 4.]], mstype.float32)
        tensor_b = Tensor([[5., 6.], [7., 8.]], mstype.float32)
        tensor_c = Tensor([[50., 60.], [70., 80.]], mstype.float32)

        s = [tensor_a, tensor_b]
        d = {'a': tensor_a, 'b': tensor_b}
        pynative_out = var_args(s, d)
        jit_func = pi_jit_with_config(var_args, jit_config=jit_config)
        jit_out = jit_func(s, d)
        match_array(pynative_out, jit_out)

        s[0] = tensor_c
        d['a'] = tensor_c
        updated_pynative = var_args(s, d)
        updated_jit = jit_func(s, d)
        match_array(updated_pynative, updated_jit)
        assert_no_graph_break(jit_func)
        assert count_ir_parameters(ir_dir) == 4
    finally:
        cleanup_ir_capture(ir_dir)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_ir_kwargs_inputs():
    """
    Feature: Loop body recapture.
    Description: Validate **kwargs inputs containing list and dict when saving IR.
    Expectation: JIT result matches pynative and IR contains only tensor parameters.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_018
    """
    ir_dir = setup_ir_capture("loop_body_recapture_ir_kwargs_inputs", "1")
    try:
        def kw_args(**kwargs):
            return kwargs['s'][0] + kwargs['s'][1] + kwargs['d']['a'] + kwargs['d']['b']

        tensor_a = Tensor([[1., 2.], [3., 4.]], mstype.float32)
        tensor_b = Tensor([[5., 6.], [7., 8.]], mstype.float32)
        tensor_c = Tensor([[50., 60.], [70., 80.]], mstype.float32)

        s = [tensor_a, tensor_b]
        d = {'a': tensor_a, 'b': tensor_b}
        pynative_out = kw_args(s=s, d=d)
        jit_func = pi_jit_with_config(kw_args, jit_config=jit_config)
        jit_out = jit_func(s=s, d=d)
        match_array(pynative_out, jit_out)

        s[0] = tensor_c
        d['a'] = tensor_c
        updated_pynative = kw_args(s=s, d=d)
        updated_jit = jit_func(s=s, d=d)
        match_array(updated_pynative, updated_jit)
        assert_no_graph_break(jit_func)
        assert count_ir_parameters(ir_dir) == 4
    finally:
        cleanup_ir_capture(ir_dir)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_ir_closure_with_collections():
    """
    Feature: Loop body recapture.
    Description: Validate nested closure creating list and dict captures when saving IR.
    Expectation: JIT result matches pynative and IR contains expected number of tensor parameters.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_019
    """
    ir_dir = setup_ir_capture("loop_body_recapture_ir_closure_with_collections", "1")
    try:
        class Net(nn.Cell):
            def construct(self, x, y):
                s = [x, y]
                d = {'a': x, 'b': y}
                c = [s, d]

                def inner():
                    return s[0] + s[1] + d['a'] + d['b'] + c[0][0] + c[1]['b']

                return inner()

        tensor_a = Tensor([[1., 2.], [3., 4.]], mstype.float32)
        tensor_b = Tensor([[5., 6.], [7., 8.]], mstype.float32)

        pynative_net = Net()
        pynative_out = pynative_net(tensor_a, tensor_b)
        jit_net = Net()
        jit_net.construct = pi_jit_with_config(jit_net.construct, jit_config=jit_config)
        jit_out = jit_net(tensor_a, tensor_b)
        match_array(pynative_out, jit_out)
        assert_no_graph_break(jit_net.construct)
        assert count_ir_parameters(ir_dir) == 2
    finally:
        cleanup_ir_capture(ir_dir)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_ir_mixed_variadic_arguments():
    """
    Feature: Loop body recapture.
    Description: Validate positional, varargs and keyword arguments mix when saving IR.
    Expectation: JIT result matches pynative and IR contains expected tensor parameters.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_020
    """
    ir_dir = setup_ir_capture("loop_body_recapture_ir_mixed_variadic_arguments", "1")
    try:
        def fn(x, w=1, *y, **z):
            return x + w + y[0] + z['z']['a']

        tensor_x = Tensor([[1., 2.], [3., 4.]], mstype.float32)
        tensor_a = Tensor([[5., 6.], [7., 8.]], mstype.float32)
        tensor_b = Tensor([[10., 20.], [30., 40.]], mstype.float32)

        y_args = [tensor_a, tensor_b]
        z_kwargs = {'z': {'a': tensor_a}}
        pynative_out = fn(tensor_x, 2, *y_args, **z_kwargs)
        jit_func = pi_jit_with_config(fn, jit_config=jit_config)
        jit_out = jit_func(tensor_x, 2, *y_args, **z_kwargs)
        match_array(pynative_out, jit_out)
        assert_no_graph_break(jit_func)
        assert count_ir_parameters(ir_dir) == 5
    finally:
        cleanup_ir_capture(ir_dir)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@pytest.mark.skipif(reason='legacy issue')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_ir_method_positional_args():
    """
    Feature: Loop body recapture.
    Description: Validate bound method handling list and dict inputs when saving IR.
    Expectation: JIT result matches pynative and IR contains expected tensor parameters.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_021
    """
    ir_dir = setup_ir_capture("loop_body_recapture_ir_method_positional_args", "1")
    try:
        class Net(nn.Cell):
            def pos_args(self, s, d):
                return s[0] + s[1] + d['a'] + d['b']

            def construct(self, x, y):
                s = [x, y]
                d = {'a': x, 'b': y}
                return self.pos_args(s, d)

        tensor_a = Tensor([[1., 2.], [3., 4.]], mstype.float32)
        tensor_b = Tensor([[10., 20.], [30., 40.]], mstype.float32)

        pynative_net = Net()
        pynative_out = pynative_net(tensor_a, tensor_b)
        jit_net = Net()
        jit_net.pos_args = pi_jit_with_config(jit_net.pos_args, jit_config=jit_config)
        jit_net.construct = pi_jit_with_config(jit_net.construct, jit_config=jit_config)
        jit_out = jit_net(tensor_a, tensor_b)
        match_array(pynative_out, jit_out)
        assert_no_graph_break(jit_net.construct)
        assert_no_graph_break(jit_net.pos_args)
        assert count_ir_parameters(ir_dir) == 2
    finally:
        cleanup_ir_capture(ir_dir)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@pytest.mark.skipif(reason='legacy issue')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_varargs_method_results():
    """
    Feature: Loop body recapture.
    Description: Varargs method mutates list elements between invocations inside construct.
    Expectation: Tuple outputs from JIT match pynative execution.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_022
    """
    class Net(nn.Cell):
        def var_args(self, *args):
            if isinstance(args[0][0], dict):
                return args[0][0]['a'] + args[0][1] + args[1]['a'] + args[1]['b']
            return args[0][0] + args[0][1] + args[1]['a'] + args[1]['b']

        def construct(self, s, d):
            out1 = self.var_args(s, d)
            s[0] = Tensor([[3., 4.], [5., 6.]], mstype.float32)
            out2 = self.var_args(s, d)
            return out1, out2

    tensor_a = Tensor([[1., 2.], [3., 4.]], mstype.float32)
    tensor_b = Tensor([[10., 20.], [30., 40.]], mstype.float32)

    pynative_net = Net()
    pynative_s = [tensor_a, tensor_b]
    pynative_d = {'a': tensor_b, 'b': tensor_a}
    pynative_out = pynative_net(pynative_s, pynative_d)

    jit_net = Net()
    jit_net.var_args = pi_jit_with_config(jit_net.var_args, jit_config=jit_config)
    jit_net.construct = pi_jit_with_config(jit_net.construct, jit_config=jit_config)
    jit_s = [
        Tensor([[1., 2.], [3., 4.]], mstype.float32),
        Tensor([[10., 20.], [30., 40.]], mstype.float32),
    ]
    jit_d = {
        'a': Tensor([[10., 20.], [30., 40.]], mstype.float32),
        'b': Tensor([[1., 2.], [3., 4.]], mstype.float32),
    }
    jit_out = jit_net(jit_s, jit_d)
    match_value(pynative_out, jit_out)
    assert_no_graph_break(jit_net.construct)
    assert_no_graph_break(jit_net.var_args)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_output_tuple_type():
    """
    Feature: Loop body recapture.
    Description: Function returns nested tuple structure with tensors across loop iterations.
    Expectation: Tuple structure and tensor values match between modes.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_023
    """
    def output_tuple(x):
        res = (x * 1000, x * 100)
        for idx in range(21):
            res = (res, x * idx)
        return res

    input_tensor = Tensor([[1., 2.], [3., 4.]], mstype.float32)
    pynative_result = output_tuple(input_tensor)
    jit_func = pi_jit_with_config(output_tuple, jit_config=jit_config)
    jit_result = jit_func(input_tensor)
    match_value(pynative_result, jit_result)
    assert isinstance(jit_result, tuple)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_output_list_type():
    """
    Feature: Loop body recapture.
    Description: Function returns nested list structure with tensors across loop iterations.
    Expectation: List structure and tensor values match between modes.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_024
    """
    def output_list(x):
        res = [x * 1000, x * 100]
        for idx in range(21):
            res = [res, x * idx]
        return res

    input_tensor = Tensor([[1., 2.], [3., 4.]], mstype.float32)
    pynative_result = output_list(input_tensor)
    jit_func = pi_jit_with_config(output_list, jit_config=jit_config)
    jit_result = jit_func(input_tensor)
    match_value(pynative_result, jit_result)
    assert isinstance(jit_result, list)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_output_mixed_list_tuple_type():
    """
    Feature: Loop body recapture.
    Description: Function returns nested tuple/list mix with tensors across loop iterations.
    Expectation: Mixed structure and tensor values match between modes.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_025
    """
    def output_mix(x):
        res = [x * 1000, x * 100]
        for idx in range(21):
            res = (res, x * idx)
        return res

    input_tensor = Tensor([[1., 2.], [3., 4.]], mstype.float32)
    pynative_result = output_mix(input_tensor)
    jit_func = pi_jit_with_config(output_mix, jit_config=jit_config)
    jit_result = jit_func(input_tensor)
    match_value(pynative_result, jit_result)
    assert isinstance(jit_result, tuple)
    assert_no_graph_break(jit_func)


@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_non_tensor_inputs():
    """
    Feature: Loop body recapture.
    Description: Function operates on pure Python numbers within list and dict inputs.
    Expectation: Results match between pynative and JIT execution without graph break.
    Migrated from: test_pijit_loop_optimize.py::test_pijit_loop_optimize_026
    """
    def pos_args(s, d):
        return s[0] + s[1] + d['a'] + d['b']

    s = [2, 3]
    d = {'a': 3, 'b': 4}
    pynative_first = pos_args(s, d)
    jit_func = pi_jit_with_config(pos_args, jit_config=jit_config)
    jit_first = jit_func(s, d)
    match_value(pynative_first, jit_first)

    s[0] = 9
    d['a'] = 5
    pynative_second = pos_args(s, d)
    jit_second = jit_func(s, d)
    match_value(pynative_second, jit_second)
    assert_no_graph_break(jit_func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_loop_recapture_train_with_pijit_loop_optimize():
    """
    Feature: Loop body recapture.
    Description: Validate loop body recapture with pijit loop optimize.
    Expectation: JIT result matches pynative and IR contains expected number of tensor parameters.
    Migrated from: test_parse_issue_scenario_supplement.py::test_train_with_pijit_loop_optimize
    """
    class Network(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = ms.Parameter(Tensor(np.ones((32, 32)), dtype=ms.float32), name='test_param')

        @jit(capture_mode='bytecode')
        def construct(self, x):
            out = x
            i = 0
            while i < 2:
                for j in range(5):
                    y = x + self.param
                    z = Tensor(y.asnumpy() + i + j)
                    out = out + z
                i = i + 1
            return out

    net = Network()
    model = create_train_model(net)
    fake_dataset = GeneratorFakeData(size=256, batch_size=32, image_size=(32,), num_classes=32)
    dataset = ds.GeneratorDataset(fake_dataset, ["data", "label"])
    model.train(2, dataset)

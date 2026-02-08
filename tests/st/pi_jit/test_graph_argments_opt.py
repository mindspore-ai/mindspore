import os
import subprocess
import shutil
import glob
import pytest
import numpy as np
from mindspore import jit, context, Tensor
from mindspore.nn import Cell
from mindspore.common import dtype as mstype
from mindspore._c_expression import update_pijit_default_config
from .share.utils import match_array, match_value, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config
from tests.st.pi_jit.one_stage.test_utils import save_graph_ir, check_ir_num

# set feature expand graph input on
config = {"expand_graph_input": True, "eliminate_redundant_args": True}


def run_sequence(s):
    return s[0] + s[3]


def run_sequence_append(s):
    s1 = []
    s1.append(s[0])
    s1.append(s[1])
    s1.append(s[2])
    s1.append(s[3])
    return s1[0] + s1[3]


def run_sequence_assign(s):
    s1 = [None, None, None, None]
    s1[0] = s[0]
    s1[1] = s[1]
    s1[2] = s[2]
    s1[3] = s[3]
    return s1[0] + s1[3]


def run_sequence_insert(s):
    s1 = []
    s1.insert(0, s[0])
    s1.insert(1, s[1])
    s1.insert(2, s[2])
    s1.insert(3, s[3])
    return s1[0] + s1[3]


def run_sequence_len(s):
    return s[0] + s[1] + len(s)


def run_sequence_pop(s):
    s.pop()
    s.pop()
    return s[0] + s[1]


def run_sequence_reverse(s):
    s.reverse()
    return s[0] + s[3]


def run_nested_sequence(s):
    s = s[0] + s[1]
    return s[0] + s[1]


def run_dict(d):
    return d.get('a') + d['d']


def run_dict_keys_values(d):
    keys = tuple(d.keys())
    values = tuple(d.values())
    d1 = {}
    for idx in range(len(keys)):
        d1[keys[idx]] = values[idx]
    return d1.get('a') + d1['d']


def run_dict_dict(d):
    d1 = dict([[key, d[key]] for key in d.keys()])
    return d1.get('a') + d1['d']


def make_dict_by_key_value(d):
    keys = tuple(d.keys())
    values = tuple(d.values())
    d1 = {}
    for idx in range(len(keys)):
        d1[keys[idx]] = values[idx]
    return d1


def make_dict_by_dict(d):
    return dict([[key, d[key]] for key in d.keys()])


def make_dict_by_items(d):
    return dict([[key, value] for key, value in d.items()])


def run_dict_sub_func(d):
    d1 = make_dict_by_dict(make_dict_by_key_value(d))
    return d1.get('a') + d1['d']


def run_dict_items(d):
    d1 = dict([[key, value] for key, value in d.items()])
    return d1.get('a') + d1['d']


def run_dict_len(d):
    return d.get('a') + d['d'] + len(d)


def run_dict_update(d):
    d1 = d.get('d1')
    d1.update(d['d2'])
    return {'ret': d1['a'] + d1.get('d')}


def run_nested_dict(d):
    d1 = make_dict_by_key_value(d.get('d1'))
    d2 = make_dict_by_dict(d['d2'])
    return {'ret': d1['a'] + d2.get('d')}


def run_nested_dict_mix(d):
    d1 = make_dict_by_dict(d.get('d1'))
    d2 = make_dict_by_items(d['d2'])
    return {'ret': d1['a'] + d2.get('d')}


def run_mix_case(s, d):
    d1 = make_dict_by_dict(d.get('d1'))
    d2 = make_dict_by_items(d['d2'])
    return s[0][0] + d1.get('a') + d2['b'][0]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize(
    'func', [run_sequence, run_sequence_append, run_sequence_assign, run_sequence_len, run_sequence_reverse]
)
@pytest.mark.parametrize('x1', [Tensor([[1.0, 2.0], [3.0, 4.0]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10.0, 20.0], [30.0, 40.0]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[100.0, 200.0], [300.0, 400.0]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[1000.0, 2000.0], [3000.0, 4000.0]], mstype.float32)])
def test_list(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand list in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = [x1, x2, y1, y2]
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    s[2] = x1
    s[3] = x2
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_sequence_pop, run_sequence_insert])
@pytest.mark.parametrize('x1', [Tensor([[1.0, 2.0], [3.0, 4.0]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10.0, 20.0], [30.0, 40.0]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[100.0, 200.0], [300.0, 400.0]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[1000.0, 2000.0], [3000.0, 4000.0]], mstype.float32)])
def test_list_break(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand list in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = [x1, x2, y1, y2]
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s)
    s1 = [x1, x2, y1, y2]
    res = func(s1)
    match_value(s, s1, error=0, err_msg=str(s))
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_nested_sequence])
@pytest.mark.parametrize('x1', [Tensor([[1.0, 2.0], [3.0, 4.0]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10.0, 20.0], [30.0, 40.0]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5.0, 6.0], [7.0, 8.0]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50.0, 60.0], [70.0, 80.0]], mstype.float32)])
def test_nest_list(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand nested list in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = [[x1, x2], [y1, y2]]
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_sequence, run_sequence_append, run_sequence_assign, run_sequence_len])
@pytest.mark.parametrize('x1', [Tensor([[1.0, 2.0], [3.0, 4.0]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10.0, 20.0], [30.0, 40.0]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5.0, 6.0], [7.0, 8.0]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50.0, 60.0], [70.0, 80.0]], mstype.float32)])
def test_tuple(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand tuple in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = (x1, x2, y1, y2)
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_nested_sequence])
@pytest.mark.parametrize('x1', [Tensor([[1.0, 2.0], [3.0, 4.0]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10.0, 20.0], [30.0, 40.0]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5.0, 6.0], [7.0, 8.0]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50.0, 60.0], [70.0, 80.0]], mstype.float32)])
def test_nested_tuple(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand nested tuple in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = ((x1, x2), (y1, y2))
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize(
    'func', [run_dict, run_dict_keys_values, run_dict_dict, run_dict_sub_func, run_dict_items, run_dict_len]
)
@pytest.mark.parametrize('x1', [Tensor([[1.0, 2.0], [3.0, 4.0]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10.0, 20.0], [30.0, 40.0]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5.0, 6.0], [7.0, 8.0]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50.0, 60.0], [70.0, 80.0]], mstype.float32)])
def test_dict(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand dict in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    d = {'a': x1, 'b': x2, 'c': y1, 'd': y2}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_nested_dict, run_nested_dict_mix])
@pytest.mark.parametrize('x1', [Tensor([[1.0, 2.0], [3.0, 4.0]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10.0, 20.0], [30.0, 40.0]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5.0, 6.0], [7.0, 8.0]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50.0, 60.0], [70.0, 80.0]], mstype.float32)])
def test_nested_dict(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand nested tuple in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    d = {'d1': {'a': x1, 'b': x2}, 'd2': {'c': y1, 'd': y2}}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(d)
    match_array(res['ret'], ms_res['ret'], error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_dict_update])
@pytest.mark.parametrize('x1', [Tensor([[1.0, 2.0], [3.0, 4.0]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10.0, 20.0], [30.0, 40.0]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5.0, 6.0], [7.0, 8.0]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50.0, 60.0], [70.0, 80.0]], mstype.float32)])
def test_nested_dict_update(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand nested tuple in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    d = {'d1': {'a': x1, 'b': x2}, 'd2': {'c': y1, 'd': y2}}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(d)
    res = func(d)
    match_array(res['ret'], ms_res['ret'], error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_mix_case])
@pytest.mark.parametrize('x1', [Tensor([[1.0, 2.0], [3.0, 4.0]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10.0, 20.0], [30.0, 40.0]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5.0, 6.0], [7.0, 8.0]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50.0, 60.0], [70.0, 80.0]], mstype.float32)])
def test_mix_case(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand vargs in function
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = [[1, x1], [3, 4, 5]]
    d = {'d1': {'a': y1, 'b': 1}, 'd2': {'a': 8, 'b': [3, 4, 5]}}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s, d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s, d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    s = [[1, x1], [6, 7, 8, 9, 10]]
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s, d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s, d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    d = {'d1': {'a': y1, 'b': 1}, 'd2': {'a': 8, 'b': [6, 7, 8]}}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s, d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s, d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


def check_ir(expect_ir_num, ir_path, expect_dict, check_shape=True):
    try:
        ir_files = sorted(glob.glob(os.path.join(ir_path, '*validate*.ir')))
        assert len(ir_files) == expect_ir_num
        if len(ir_files) == 2:
            file = ir_files[1]
        elif len(ir_files) == 1:
            file = ir_files[0]
        else:
            return

        for key in expect_dict:
            cmd = f"grep '^%para' {file} | grep '{key}' | wc -l"
            output = subprocess.check_output(cmd, shell=True)
            output = str(output, 'utf-8').strip()
            assert int(output) == expect_dict[key]
            if check_shape:
                cmd2 = f"grep '^{key}' {file} | wc -l"
                output2 = subprocess.check_output(cmd2, shell=True)
                output2 = str(output2, 'utf-8').strip()
                assert int(output2) == expect_dict[key]

    finally:
        if 'MS_DEV_SAVE_GRPHS' in os.environ:
            del os.environ['MS_DEV_SAVE_GRAPHS']
        if 'MS_DEV_SAVE_GRAPHS_PATH' in os.environ:
            del os.environ['MS_DEV_SAVE_GRAPHS_PATH']
        shutil.rmtree(ir_path)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_pos_arg_unused_in_subfunc():
    """
    Feature: Graph parameter elimination.
    Description: Test that an unused positional argument passed to a sub-function is eliminated from the graph.
    Expectation: The result is correct and IR is as expected.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_001
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.t = 2

        def construct(self, x, y, z):
            if x > 1:
                out = self.func1(y, z)
            else:
                out = z
            return out

        def func1(self, a, b):
            b = b + b
            return self.t * b

    case_name = "test_param_elimination_pos_arg_unused_in_subfunc"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    x = Tensor(np.array(2).astype(np.float32))
    y = Tensor(np.random.rand(2, 3).astype(np.float32))
    z = Tensor(np.random.rand(4, 5).astype(np.float32))

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)
    out = net(x, y, z)
    pynative_out = net(x, y, z)
    match_array(out, pynative_out)
    check_ir(2, ir_path, {"%para2": 0, "(2, 3)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_pos_arg_unused_in_toplevel():
    """
    Feature: Graph parameter elimination.
    Description: Test that an unused positional argument in the top-level function is eliminated from the graph.
    Expectation: The result is correct and IR is as expected.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_002
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.t = 2

        def construct(self, x, y, z):
            if x > 1:
                out = z + z
            else:
                out = z
            return out

    case_name = "test_param_elimination_pos_arg_unused_in_toplevel"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    x = Tensor(np.array(2).astype(np.float32))
    y = Tensor(np.random.rand(2, 3).astype(np.float32))
    z = Tensor(np.random.rand(4, 5).astype(np.float32))

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)
    out = net(x, y, z)
    pynative_out = Net()(x, y, z)
    match_array(out, pynative_out)
    check_ir(2, ir_path, {"%para2": 0, "(2, 3)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_multiple_pos_args_unused():
    """
    Feature: Graph parameter elimination.
    Description: Test that multiple unused positional arguments are eliminated from the graph.
    Expectation: The result is correct and IR is as expected.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_003
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.t = 2
        def construct(self, x, y, z):
            return x * x * x * self.t

    case_name = "test_param_elimination_multiple_pos_args_unused"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    x = Tensor(np.random.rand(1, 2).astype(np.float32))
    y = Tensor(np.random.rand(3, 4).astype(np.float32))
    z = Tensor(np.random.rand(5, 6).astype(np.float32))
    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)
    out = net(x, y, z)
    pynative_out = Net()(x, y, z)
    match_array(out, pynative_out)
    check_ir(1, ir_path, {"%para2": 0, "(3, 4)": 0, "(5, 6)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_multiple_pos_args_unused_in_subfunc():
    """
    Feature: Graph parameter elimination.
    Description: Test that multiple unused positional arguments in a sub-function are eliminated.
    Expectation: The result is correct and IR is as expected.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_004
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.t = 2

        def construct(self, x, y, z, m, n):
            if x > 1:
                out = self.func1(y, z, m, n)
            else:
                out = self.func1(z, y, z, y)
            return out

        def func1(self, a, b, c, d):
            return self.t * a + 3 * d

    case_name = "test_param_elimination_multiple_pos_args_unused_in_subfunc"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    x = Tensor(np.array(2).astype(np.float32))
    y = Tensor(np.random.rand(2, 3).astype(np.float32))
    z = Tensor(np.random.rand(2, 1).astype(np.float32))
    m = Tensor(np.random.rand(2, 4).astype(np.float32))
    n = Tensor(np.random.rand(2, 3).astype(np.float32))

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)
    out = net(x, y, z, m, n)
    pynative_out = Net()(x, y, z, m, n)
    match_array(out, pynative_out)
    check_ir(2, ir_path, {"%para3": 0, "(2, 1)": 0, "(2, 4)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_list_unused_element_shape_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for list input where an unused element changes shape between calls.
    Expectation: The result is correct, and only one graph is compiled.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_005
    """
    class Net(Cell):
        def construct(self, x):
            out = x[0] + x[0]
            return out

    case_name = "test_param_elimination_list_unused_element_shape_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    x1 = [Tensor([1, 1]), Tensor([2, 2, 2])]
    out1 = net(x1)
    match_array(out1, Tensor([2, 2]))

    x2 = [Tensor([3, 3]), Tensor([2, 2, 2, 3])]
    out2 = net(x2)
    match_array(out2, Tensor([6, 6]))
    check_ir(1, ir_path, {"%para2": 0, "(4)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_list_used_element_shape_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for list input where a used element changes shape, causing recompilation.
    Expectation: The result is correct.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_006
    """
    class Net(Cell):
        def construct(self, x):
            out = x[0] + x[0]
            return out

    case_name = "test_param_elimination_list_used_element_shape_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    x1 = [Tensor([1, 1]), Tensor([2, 2, 2, 2])]
    out1 = net(x1)
    match_array(out1, Tensor([2, 2]))

    x2 = [Tensor([3, 3, 3]), Tensor([2, 2, 2, 2])]
    out2 = net(x2)
    match_array(out2, Tensor([6, 6, 6]))
    check_ir(2, ir_path, {"%para2": 0, "(4)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_list_unused_element_len_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for list input where the list length changes.
    Expectation: The result is correct.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_007
    """
    class Net(Cell):
        def construct(self, x):
            out = x[0] + x[0]
            return out

    case_name = "test_param_elimination_list_unused_element_len_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    x1 = [Tensor([1, 1]), Tensor([2, 2, 2])]
    out1 = net(x1)
    match_array(out1, Tensor([2, 2]))
    x2 = [Tensor([1, 1]), Tensor([2, 2, 2]), Tensor([3, 3, 3])]
    out2 = net(x2)
    match_array(out2, Tensor([2, 2]))
    check_ir(2, ir_path, {"%para2": 0, "(3)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_tuple_unused_element_value_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for tuple input where an unused non-tensor element changes value.
    Expectation: The result is correct, and only one graph is compiled.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_008
    """
    class Net(Cell):
        def construct(self, x):
            out = x[1] + x[0]
            return out

    case_name = "test_param_elimination_tuple_unused_element_value_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    x1 = (Tensor([1, 1]), Tensor([2, 2]), 3)
    out1 = net(x1)
    match_array(out1, Tensor([3, 3]))
    x2 = (Tensor([3, 3]), Tensor([2, 2]), 1)
    out2 = net(x2)
    match_array(out2, Tensor([5, 5]))
    check_ir(1, ir_path, {"%para3": 0}, False)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_tuple_len_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for tuple input where the tuple length changes.
    Expectation: The result is correct.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_009
    """
    class Net(Cell):
        def construct(self, x):
            out = x[1] + x[0]
            return out

    case_name = "test_param_elimination_tuple_len_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    x1 = (Tensor([1, 1]), Tensor([2, 2]), 3, 4)
    out1 = net(x1)
    match_array(out1, Tensor([3, 3]))
    x2 = (Tensor([1, 1]), Tensor([2, 2]), Tensor([3, 3, 3]))
    out2 = net(x2)
    match_array(out2, Tensor([3, 3]))
    check_ir(2, ir_path, {"%para3": 0, "(3)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_dict_unused_item_shape_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for dict input where an unused item changes shape.
    Expectation: The result is correct, and only one graph is compiled.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_010
    """
    class Net(Cell):
        def construct(self, x):
            out = x['a'] + x['b']
            return out

    case_name = "test_param_elimination_dict_unused_item_shape_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    x1 = {'a': Tensor([1, 1]), 'b': Tensor([2, 2]), 'c': Tensor([3, 3, 3])}
    out1 = net(x1)
    match_array(out1, Tensor([3, 3]))
    x2 = {'a': Tensor([1, 1]), 'b': Tensor([2, 2]), 'c': Tensor([5, 5, 5, 5])}
    out2 = net(x2)
    match_array(out2, Tensor([3, 3]))
    check_ir(1, ir_path, {"%para3": 0, "(3)": 0, "(4)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_dict_len_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for dict input where the dict length changes.
    Expectation: The result is correct, and only one graph is compiled.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_011
    """
    class Net(Cell):
        def construct(self, x):
            out = x['a'] + x['b']
            return out

    case_name = "test_param_elimination_dict_len_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    x1 = {'a': Tensor([1, 1]), 'b': Tensor([2, 2]), 'c': Tensor([3, 3])}
    out1 = net(x1)
    match_array(out1, Tensor([3, 3]))
    x2 = {'a': Tensor([1, 1]), 'b': Tensor([2, 2]), 'c': Tensor([3, 3]), 'd': 4}
    out2 = net(x2)
    match_array(out2, Tensor([3, 3]))
    check_ir(1, ir_path, {"%para3": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_nested_list_shape_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for nested list input where unused elements change.
    Expectation: The result is correct.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_012
    """
    class Net(Cell):
        def construct(self, x):
            out = x[0][0] + x[1]
            return out

    case_name = "test_param_elimination_nested_list_shape_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    x1 = [[Tensor([1, 1]), Tensor([2, 2, 2])], Tensor([3, 3])]
    out1 = net(x1)
    match_array(out1, Tensor([4, 4]))
    x2 = [[Tensor([3, 3]), Tensor([5, 5, 5])], Tensor([4, 4]), Tensor([3, 5, 4, 5])]
    out2 = net(x2)
    match_array(out2, Tensor([7, 7]))
    check_ir(2, ir_path, {"%para3": 0, "(3)": 0, "(4)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_vargs_shape_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for variable arguments (*args) where an unused arg changes shape.
    Expectation: The result is correct, and only one graph is compiled.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_013
    """
    class Net(Cell):
        def construct(self, *x):
            out = x[0] + x[0]
            return out

    case_name = "test_param_elimination_vargs_shape_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    out1 = net(Tensor([1, 1]), Tensor([2, 2]))
    match_array(out1, Tensor([2, 2]))
    out2 = net(Tensor([3, 3]), Tensor([2, 2, 2]))
    match_array(out2, Tensor([6, 6]))
    check_ir(1, ir_path, {"%para2": 0, "(3)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_vargs_len_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for variable arguments (*args) where the number of arguments changes.
    Expectation: The result is correct.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_014
    """
    class Net(Cell):
        def construct(self, *x):
            out = x[0] + x[0]
            return out

    case_name = "test_param_elimination_vargs_len_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    out1 = net(Tensor([1, 1]), Tensor([2, 2]))
    match_array(out1, Tensor([2, 2]))
    out2 = net(Tensor([3, 3]), Tensor([2, 2]), Tensor([3, 3, 3]))
    match_array(out2, Tensor([6, 6]))
    check_ir(2, ir_path, {"%para2": 0, "(3)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_kwargs_shape_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for keyword arguments (**kwargs) where an unused item changes shape.
    Expectation: The result is correct, and only one graph is compiled.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_015
    """
    class Net(Cell):
        def construct(self, x, **d):
            out = x * d['a'] + d['b']
            return out

    case_name = "test_param_elimination_kwargs_shape_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    x1 = Tensor([3, 3])
    d1 = {"a": Tensor([1, 1]), "b": Tensor([2, 2]), "c": Tensor([3, 3])}
    out1 = net(x1, **d1)
    match_array(out1, Tensor([5, 5]))

    x2 = Tensor([4, 4])
    d2 = {"a": Tensor([4, 4]), "b": Tensor([2, 2]), "c": Tensor([5, 5, 5])}
    out2 = net(x2, **d2)
    match_array(out2, Tensor([18, 18]))
    check_ir(1, ir_path, {"%para4": 0, "(3)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_kwargs_len_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for keyword arguments (**kwargs) where the number of items changes.
    Expectation: The result is correct, and only one graph is compiled.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_016
    """
    class Net(Cell):
        def construct(self, x, **d):
            out = x * d['a'] + d['b']
            return out

    case_name = "test_param_elimination_kwargs_len_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    x1 = Tensor([3, 3])
    d1 = {"a": Tensor([1, 1]), "b": Tensor([2, 2]), "c": Tensor([3, 3, 3])}
    out1 = net(x1, **d1)
    match_array(out1, Tensor([5, 5]))

    x2 = Tensor([4, 4])
    d2 = {"a": Tensor([4, 4]), "b": Tensor([2, 2]), "c": Tensor([5, 5, 5]), "d": 4}
    out2 = net(x2, **d2)
    match_array(out2, Tensor([18, 18]))
    check_ir(1, ir_path, {"%para4": 0, "(3)": 0})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_param_elimination_mixed_args_change():
    """
    Feature: Graph parameter elimination.
    Description: Test parameter elimination for a mix of argument types with changing shapes and lengths.
    Expectation: The result is correct.
    Migrated from: test_parse_pijit_parameter_elimination.py::test_parse_pijit_parameter_elimination_017
    """
    class Net(Cell):
        def construct(self, a, b, c=3, *d, **e):
            out = a * c + d[0] * e['x']
            return out

    case_name = "test_param_elimination_mixed_args_change"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path

    net = Net()
    net.construct = pi_jit_with_config(net.construct, jit_config=config)

    a1 = Tensor([1, 1])
    b1 = 2
    d1 = [Tensor([3, 3]), 3]
    e1 = {'x': Tensor([4, 4]), 'y': Tensor([5, 5, 5, 5])}
    out1 = net(a1, b1, 3, *d1, **e1)
    match_array(out1, Tensor([15, 15]))

    a2 = Tensor([6, 6])
    b2 = Tensor([2, 2, 2])
    d2 = [Tensor([7, 7]), 3, 4]
    e2 = {'x': Tensor([8, 8]), 'z': Tensor([9, 9, 5, 5])}
    out2 = net(a2, b2, 3, *d2, **e2)
    match_array(out2, Tensor([74, 74]))
    check_ir(2, ir_path, {"%para4": 0, "(3)": 0, "(4)": 0}, False)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_elimination_01():
    """
    Feature: parameter elimination
    Description: test cases for parameter elimination
    Expectation: The result match and no exception
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.x = 1

        @jit(capture_mode="bytecode", backend="ms_backend")
        def construct(self, x):
            if self.x > 3:
                self.x = x
            else:
                self.x = self.x + 1
            return self.x + x

    update_pijit_default_config(eliminate_redundant_args=True)
    net = Net()
    expected_y = [2, 4, 6, 6, 8, 10, 12, 14, 16, 18]
    actual_y = []
    for i in range(10):
        x = Tensor([i])
        y = net(x)
        actual_y.append(y.asnumpy()[0])

    assert actual_y == expected_y
    check_ir_num('graph_before_compile', 8)


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_elimination_02():
    """
    Feature: parameter elimination
    Description: test cases for parameter elimination
    Expectation: The result match and no exception
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.x = 1

        @jit(capture_mode="bytecode", backend="ms_backend")
        def construct(self, x):
            if self.x > 3:
                self.x = x
            else:
                self.x = self.x + 1
            return self.x + x

    update_pijit_default_config(eliminate_redundant_args=False)
    net = Net()
    expected_y = [2, 4, 6, 6, 8, 10, 12, 14, 16, 18]
    actual_y = []
    for i in range(10):
        x = Tensor([i])
        y = net(x)
        actual_y.append(y.asnumpy()[0])

    assert actual_y == expected_y
    check_ir_num('graph_before_compile', 8)

import pytest
from mindspore import jit, context, Tensor
from mindspore.common import dtype as mstype
from .share.utils import match_array, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config

# set feature expand graph input on
config = { "expand_graph_input": True, }

def run_sequence(s):
    return s[0] + s[1] + s[2] + s[3]

def run_nested_sequence(s):
    return s[0][0] + s[0][1] + s[1][0] + s[1][1]

def run_mix_nest_list_tuple_dict(s):
    return s[0][1] + s[0][1] + s[1]['a'] + s[1]['b']

def run_dict(d):
    return d['a'] + d['b'] + d['c'] + d['d']

def run_nested_dict(d):
    return d['d1']['a'] + d['d1']['b'] + d['d2']['c'] + d['d2']['d']

def run_mix_nest_dict_list_tuple(d):
    return d['l'][0] + d['l'][1] + d['t'][0] + d['t'][1]

def run_vargs_1(*args):
    return args[0] + args[1] + args[2] + args[3]

def run_vargs_2(*args):
    return args[0] + args[1][0] + args[1][1][0] + args[1][1][1]

def run_vargs_3(*args):
    return args[0] + args[1][0]['a'] + args[1][1][0] + args[1][1][1]

def run_kwargs_1(**kwargs):
    return kwargs['k'] + kwargs['s'][0][0] + kwargs['s'][0][1] + kwargs['d']['a']

def run_kwargs_2(**kwargs):
    return kwargs['k'] + kwargs['s'][0]['a'][0]['b'] +  kwargs['s'][1][0]['c'] + kwargs['d']['e'][0]

def run_mix_args_vargs_kwargs(pos, *args, **kwargs):
    return pos['l'][0] + pos['l'][1] + pos['t'][0] + pos['t'][1] + args[0] + args[1][0]['a'] + args[1][1][0] + \
           args[1][1][1] + kwargs['k'] + kwargs['s'][0]['a'][0]['b'] +  kwargs['s'][1][0]['c'] + kwargs['d']['e'][0]

def run_closure_1(a, b, c, d):
    s = [a, b, c, d]
    def inner():
        return s[0] + s[1] + s[2] + s[3]
    return inner

def run_closure_2(a, b, c, d):
    s = [(a, b), {'a': c, 'b': d}]
    def inner():
        return s[0][1] + s[0][1] + s[1]['a'] + s[1]['b']
    return inner

def run_closure_3(a, b, c, d):
    s = {'t': (a, b), 'l': [c, d]}
    def inner():
        return s['l'][0] + s['l'][1] + s['t'][0] + s['t'][1]
    return inner

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_sequence])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
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
    s[0] = x1 * 10
    s[1] = x2 * 10
    s[2] = y1 * 10
    s[3] = y2 * 10
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_nested_sequence])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
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
    s[0][0] = x1 * 10
    s[0][1] = x2 * 10
    s[1][0] = y1 * 10
    s[1][1] = y2 * 10
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_sequence])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
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
    s = (x1 * 10, x2 * 10, y1 * 10, y2 * 10)
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_nested_sequence])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
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
    s = ((x1 * 10, x2 * 10), (y1 * 10, y2 * 10))
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_nested_sequence])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_mix_nested_list_tuple(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand nested tuple in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = ([x1, x2], (y1, y2))
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    s = ([x1 * 10, x2 * 10], (y1 * 10, y2 * 10))
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_mix_nest_list_tuple_dict])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_mix_nested_list_tuple_dict(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand mix nested list/tuple/dict in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = [(x1, x2), {'a': y1, 'b': y2}]
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    s = [(x1 * 10, x2 * 10), {'a': y1 * 10, 'b': y2 * 10}]
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_dict])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_dict(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand dict in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    d = {'a': x1, 'b': x2, 'c': y1, 'd':y2}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    d['a'] = x1 * 10
    d['b'] = x2 * 10
    d['c'] = y1 * 10
    d['d'] = y2 * 10
    ms_res = wrapped_func(d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_nested_dict])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_nested_dict(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand nested tuple in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    d = { 'd1': {'a': x1, 'b': x2}, 'd2': {'c': y1, 'd':y2}}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    d['d1']['a'] = x1 * 10
    d['d1']['b'] = x2 * 10
    d['d2']['c'] = y1 * 10
    d['d2']['d'] = y2 * 10
    ms_res = wrapped_func(d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_mix_nest_dict_list_tuple])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_mix_nested_dict_list_tuple(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand mix nested list/tuple/dict in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    d = {'t': (x1, x2), 'l': [y1, y2]}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    d = {'t': (x1 * 10, x2 * 10), 'l': [y1 * 10, y2 * 10]}
    ms_res = wrapped_func(d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_vargs_1])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_vargs_1(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand vargs in function
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(x1, x2, y1, y2)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(x1, x2, y1, y2)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    ms_res = wrapped_func(x1 * 10, x2 * 10, y1 * 10, y2 * 10)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(x1 * 10, x2 * 10, y1 * 10, y2 * 10)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_vargs_2])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_vargs_2(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand vargs in function
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = (x2, [y1, y2])
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(x1, s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(x1, s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    s = (x2 * 10, [y1 * 10, y2 * 10])
    ms_res = wrapped_func(x1 * 10, s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(x1 * 10, s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_vargs_3])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_vargs_3(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand vargs in function
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = [{'a': x2}, (y1, y2)]
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(x1, s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(x1, s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    s = [{'a': x2 * 10}, (y1 * 10, y2 * 10)]
    ms_res = wrapped_func(x1 * 10, s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(x1 * 10, s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_kwargs_1])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_kwargs_1(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand vargs in function
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(k=x1, s=[x2, y1], d={'a': y2})
    assert_executed_by_graph_mode(wrapped_func)
    res = func(k=x1, s=[x2, y1], d={'a': y2})
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    ms_res = wrapped_func(k=x1 * 10, s=[x2 * 10, y1 * 10], d={'a': y2 *10})
    assert_executed_by_graph_mode(wrapped_func)
    res = func(k=x1 * 10, s=[x2 * 10, y1 * 10], d={'a': y2 *10})
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_kwargs_2])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_kwargs_2(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand vargs in function
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(k=x1, s=({'a':[{'b': x2}]}, [{'c': y1}]), d={'e': [y2]})
    assert_executed_by_graph_mode(wrapped_func)
    res = func(k=x1, s=({'a':[{'b': x2}]}, [{'c': y1}]), d={'e': [y2]})
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    ms_res = wrapped_func(k=x1 * 10, s=({'a':[{'b': x2 * 10}]}, [{'c': y1 * 10}]), d={'e': [y2 * 10]})
    assert_executed_by_graph_mode(wrapped_func)
    res = func(k=x1 * 10, s=({'a':[{'b': x2 * 10}]}, [{'c': y1 * 10}]), d={'e': [y2 * 10]})
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_mix_args_vargs_kwargs])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_mix_args_vargs_kwargs(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand vargs in function
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    d = {'t': (x1, x2), 'l': [y1, y2]}
    s = [{'a': x2}, (y1, y2)]
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(d, x1, s, k=x1, s=({'a':[{'b': x2}]}, [{'c': y1}]), d={'e': [y2]})
    assert_executed_by_graph_mode(wrapped_func)
    res = func(d, x1, s, k=x1, s=({'a':[{'b': x2}]}, [{'c': y1}]), d={'e': [y2]})
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    d = {'t': (x1 * 10, x2 * 10), 'l': [y1 * 10, y2 * 10]}
    s = [{'a': x2 * 10}, (y1 * 10, y2 * 10)]
    ms_res = wrapped_func(d, x1 * 10, s, k=x1 * 10, s=({'a':[{'b': x2 * 10}]}, [{'c': y1 * 10}]), d={'e': [y2 * 10]})
    assert_executed_by_graph_mode(wrapped_func)
    res = func(d, x1 * 10, s, k=x1 * 10, s=({'a':[{'b': x2 * 10}]}, [{'c': y1 * 10}]), d={'e': [y2 * 10]})
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_closure_1])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_closure_1(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand vargs in function
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    inner = func(x1, x2, y1, y2)
    wrapped_func = pi_jit_with_config(inner, jit_config=config)
    ms_res = wrapped_func()
    assert_executed_by_graph_mode(wrapped_func)
    res = inner()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_closure_2])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_closure_2(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand vargs in function
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    inner = func(x1, x2, y1, y2)
    wrapped_func = pi_jit_with_config(inner, jit_config=config)
    ms_res = wrapped_func()
    assert_executed_by_graph_mode(wrapped_func)
    res = inner()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('func', [run_closure_3])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_closure_3(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand vargs in function
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    inner = func(x1, x2, y1, y2)
    wrapped_func = pi_jit_with_config(inner, jit_config=config)
    ms_res = wrapped_func()
    assert_executed_by_graph_mode(wrapped_func)
    res = inner()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

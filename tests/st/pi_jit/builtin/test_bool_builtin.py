import pytest
import numpy as onp
from mindspore import Tensor, jit, context
from ..share.utils import match_array


@jit(mode="PIJit")
def fallback_bool_empty():
    res = bool()
    return res


@jit(mode="PIJit")
def fallback_bool_int():
    res = bool(int)
    return res


@jit(mode="PIJit")
def fallback_bool_zero():
    res = bool(0)
    return res


@jit(mode="PIJit")
def fallback_bool_no_zero():
    res = bool(1)
    return res


@jit(mode="PIJit")
def fallback_bool_list():
    res = bool([1, 2, 3, 4])
    return res


@jit(mode="PIJit")
def fallback_bool_tuple():
    res = bool((1, 2))
    return res


@jit(mode="PIJit")
def fallback_bool_empty_list():
    res = bool([])
    return res


@jit(mode="PIJit")
def fallback_bool_empty_tuple():
    res = bool(tuple())
    return res


@jit(mode="PIJit")
def fallback_bool_str():
    res = bool("123")
    return res


@jit(mode="PIJit")
def fallback_bool_numpy(a):
    res = bool(a)
    return res


@jit(mode="PIJit")
def fallback_bool_complex():
    res1 = bool(complex(0, 0))
    res2 = bool(complex(1, 0))
    res3 = bool(complex(0, 1))
    return res1, res2, res3


@jit(mode="PIJit")
def fallback_bool_none():
    res = bool(None)
    return res


@jit(mode="PIJit")
def fallback_bool_tensor(a, b, c, d):
    res1 = bool(a)
    res2 = bool(b)
    res3 = bool(c)
    res4 = bool(d)
    return res1, res2, res3, res4


@jit
def ms_fallback_bool_empty():
    res = bool()
    return res


@jit
def ms_fallback_bool_int():
    res = bool(int)
    return res


@jit
def ms_fallback_bool_zero():
    res = bool(0)
    return res


@jit
def ms_fallback_bool_no_zero():
    res = bool(1)
    return res


@jit
def ms_fallback_bool_list():
    res = bool([1, 2, 3, 4])
    return res


@jit
def ms_fallback_bool_tuple():
    res = bool((1, 2))
    return res


@jit
def ms_fallback_bool_empty_list():
    res = bool([])
    return res


@jit
def ms_fallback_bool_empty_tuple():
    res = bool(tuple())
    return res


@jit
def ms_fallback_bool_str():
    res = bool("123")
    return res


@jit
def ms_fallback_bool_numpy():
    res = bool(onp.array(1.5))
    return res


@jit
def ms_fallback_bool_complex():
    res1 = bool(complex(0, 0))
    res2 = bool(complex(1, 0))
    res3 = bool(complex(0, 1))
    return res1, res2, res3


@jit
def ms_fallback_bool_none():
    res = bool(None)
    return res


@jit
def ms_fallback_bool_tensor(a, b, c, d):
    res1 = bool(a)
    res2 = bool(b)
    res3 = bool(c)
    res4 = bool(d)
    return res1, res2, res3, res4


test_data = [
    (fallback_bool_empty, ms_fallback_bool_empty, [], {}),
    (fallback_bool_int, ms_fallback_bool_int, [], {}),
    (fallback_bool_zero, ms_fallback_bool_zero, [], {}),
    (fallback_bool_list, ms_fallback_bool_list, [], {}),
    (fallback_bool_tuple, ms_fallback_bool_tuple, [], {}),
    (fallback_bool_empty_list, ms_fallback_bool_empty_list, [], {}),
    (fallback_bool_str, ms_fallback_bool_str, [], {}),
    (fallback_bool_numpy, ms_fallback_bool_numpy, [onp.array(1.5)], {}),
    (fallback_bool_complex, ms_fallback_bool_complex, [], {}),
    (fallback_bool_none, ms_fallback_bool_none, [], {}),
]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func,ms_func,args,kwargs', test_data)
def test_bool(func, ms_func, args, kwargs):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    Test Steps:
    1. Test bool() in PYNATIVE mode with test_data
    2. output bool
     """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(*args, **kwargs)
    context.set_context(mode=context.GRAPH_MODE)
    if ms_func is ms_fallback_bool_numpy:
        ms_res = ms_func()
    else:
        ms_res = ms_func(*args, **kwargs)

    if isinstance(res, tuple):
        for r, ms_r in zip(res, ms_res):
            match_array(r, ms_r, error=0, err_msg=str(ms_r))
    else:
        match_array(res, ms_res, error=0, err_msg=str(ms_res))


tensor_test_data = [
    (fallback_bool_tensor, ms_fallback_bool_tensor,
     [Tensor([1]), Tensor([0]), (Tensor(1), Tensor(2)), (Tensor(0), Tensor(0))], {}),
]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func,ms_func,args,kwargs', tensor_test_data)
def test_bool_tensor(func, ms_func, args, kwargs):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    Test Steps:
    1. Test bool() in PYNATIVE mode with tensor
    2. output bool
     """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(*args, **kwargs)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(*args, **kwargs)

    if isinstance(res, tuple):
        for r, ms_r in zip(res, ms_res):
            match_array(r, ms_r, error=0, err_msg=str(ms_r))
    else:
        match_array(res, ms_res, error=0, err_msg=str(ms_res))

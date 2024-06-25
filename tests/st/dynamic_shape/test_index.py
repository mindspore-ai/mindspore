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

'''test tensor index'''

import mindspore as ms
from mindspore import Tensor, ops, context, nn
from mindspore.ops.composite import GradOperation
import numpy as np
import sys
import pytest
from tests.mark_utils import arg_mark

TEST_LIST = []
set_exclusive = False

def _add_testcase_to_testlist(exclusive=False):
    """_add_testcase_to_testlist"""
    def add(fn):
        global set_exclusive
        if exclusive and not set_exclusive:
            set_exclusive = True
            TEST_LIST.clear()
            TEST_LIST.append(fn)
        elif not set_exclusive:
            TEST_LIST.append(fn)

    return add


# setitem


@_add_testcase_to_testlist()
def _run_setitem_by_slice1(data):
    """test index"""
    data_new0 = data + 0
    data_new0[1:3:1] = 4

    data_new1 = data + 0
    data_new1[Tensor(1):3:1] = 5

    data_new2 = data + 0
    data_new2[True, True] = 4

    return data_new0, data_new1, data_new2


@_add_testcase_to_testlist()
def _run_setitem_by_number(data):
    """test index"""
    out0 = data + 0
    out0[1] = 102

    # setitem with Number
    out0[data.shape[0] - 1] = data.shape[0]
    out0[0][data.shape[1] - 1] = 100

    # setitem with Tensor
    out0[data.shape[0] - 2] = Tensor([101, 101, 101, 101, 101])

    out1 = data + 0

    # setitem with sequence
    out1[data.shape[0] - 1] = (103, 103, 103, 103, 103)
    out1[data.shape[0] - 2] = [104, data.shape[1], 104, 104, 104]
    out1[data.shape[0] - 3] = (105, Tensor(105), 105.0, data[0][1], 105)

    return out0, out1


@_add_testcase_to_testlist()
def _run_setitem_by_ellipsis(data):
    """test index"""
    # setitem with Number
    out0 = data + 0
    out0[...] = data.shape[1]

    # setitem with Tensor
    out1 = data + 0
    out1[...] = data[0]

    # setitem with sequence
    out2 = data + 0
    out2[...] = (104, data.shape[1], 104, 104, 104)
    return out0, out1, out2


@_add_testcase_to_testlist()
def _run_setitem_by_ellipsis_4d(data):
    """test index"""
    out0 = data + 0
    out0[..., ::2, 1::2] = data[0][1][1][1]

    # failed in pynative
    # out1 = data + 0
    # out1[..., ::2, ::-1] = [111, 112, 113, 114]

    # failed in pynative
    # out2 = data + 0
    # out2[1::-1, ..., 2::-1] = [111, 112]

    # failed in pynative
    # out3 = data + 0
    # out3[1::-1, ::-2, ...] = [111, 112, 113, 114]
    # return out3

    out4 = data + 0
    out4[::, ..., ::2] = [111, 112]

    out5 = data + 0
    out5[1::-1, :4:-2, ...] = [111, 112, 113, 114]
    return out0, out4, out5


@_add_testcase_to_testlist()
def _run_setitem_by_tensor(data):
    """test index"""
    # setitem by int Tensor with Number
    out0 = data + 0
    out0[Tensor(1)] = 1
    out0[Tensor([2, 3])] = 108

    # setitem by True Tensor with Number
    out1 = data + 0
    out1[Tensor(True)] = 2

    # setitem by False Tensor with Number
    out2 = data + 0
    out2[Tensor(False)] = 3
    out2[Tensor([True, False, True, False])] = 103

    # setitem by int Tensor with sequence
    # Failed in Pynative mode
    # out3 = data + 0
    # out3[Tensor(1)] = [104, data.shape[1], 105, 106, 107]
    return out0, out1, out2


@_add_testcase_to_testlist()
def _run_setitem_by_slice(data):
    """test index"""
    out0 = data + 0
    # setitem by slice(Number) with Number
    out0[1:2:1] = 101
    out0[data.shape[0] - 2:data.shape[0] - 1:1] = 102
    # setitem by slice(Tensor) with Number
    out0[data[0][0]:data[0][1]:data[0][1]] = 100

    out1 = data + 0
    # setitem by slice with Tensor
    out1[1:2:1] = data[3]
    #setitem by slice with sequence
    out1[2:3:1] = [data.shape[0], data.shape[1], Tensor(1), 100, 100]

    # negative index in slice
    out2 = data + 0
    out2[data.shape[0] - 5: data.shape[0] - 7:data[0][0] - 1] = 111

    return out0, out1, out2


@_add_testcase_to_testlist()
def _run_setitem_by_sequence(data):
    """test index"""
    out0 = data + 0
    # setitem by tuple with Number
    out0[0, 2:4:1] = 104
    out0[data.shape[0] - 3, 2:4:1] = 104

    # setitem by tuple with Tensor
    out0[data.shape[0] - 2, data[0][0]:data.shape[1]:data[0][1]] = data[3][0]
    out0[data.shape[0] - 1, data[0][0]:data.shape[1]:1] = data[0]

    # setitem by tuple with list
    out0[data.shape[0] - 1, data[0][0]:data.shape[1] - 2:] = [106, 107, data[3][4]]

    # setitem by tuple(has ellipsis)
    out0[data.shape[0] - 2, ...] = data[0]

    # setitem by list
    out1 = data + 0
    out1[[data.shape[0] - 2, 0]] = data[3]
    out1[[data.shape[0] - 1, 0]] = 199
    out1[[data.shape[0] - 3, 3]] = [104, 105, 106, data[0][1], data.shape[1]]

    out2 = data + 0
    out2[[True, False, False, True]] = 199
    out2[[[data.shape[0] - 1, 2], 1]] = 200


    # negative index in slice
    out3 = data + 0
    out3[1, -1:data.shape[1] - 1:1] = 111
    out3[3, -1:data.shape[1] - 7:-1] = 112
    return out0, out1, out2, out3


### getitem

@_add_testcase_to_testlist()
def _run_getitem_by_slice(data):
    """test index"""
    out1 = data[3:2:1]
    out2 = data[data.shape[0] - 3:data.shape[0] - 1:data[0][1]]
    out3 = data[:]
    out4 = data[:-1]
    out5 = data[None:None:1]

    out6 = data[data.shape[0] - 6:data.shape[0] - 1:data[0][0] - 1]
    out7 = data[data.shape[0] - 6:data.shape[0] - 1:data[0][1]]

    return out1, out2, out3, out4, out5, out6, out7


@_add_testcase_to_testlist()
def _run_getitem_by_tensor(data):
    """test index"""
    # getitem by Tensor(1)
    out1 = data[data[0][1]]

    # getitem by Tensor([[1], [2]])
    index = data[0][1:3:1]
    index = ops.reshape(index, (2, 1))
    out2 = data[data[0][1:3:1]]

    out3 = data[Tensor(np.array([[1, 2], [0, 1]]).astype(np.int64))]
    return out1, out2, out3


@_add_testcase_to_testlist()
def _run_getitem_by_list(data):
    """test index"""
    out0 = data[[True, True, False, True]]
    out1 = data[[1, data.shape[1] - 2]]
    out2 = data[[1, True, data[0][2]]]
    return out0, out1, out2


@_add_testcase_to_testlist()
def _run_getitem_by_tuple(data):
    """test index"""
    out0 = data[data.shape[0] - 2, 4]
    out1 = data[data[0][1], data[0][0]:data[0][2]]
    out2 = data[Tensor([[1, 2], [2, 3]]), 1:]
    out3 = data[Tensor([[1, 2], [2, 3]]), 1:0]
    out4 = data[True, 1]
    out5 = data[:True:, True]
    out6 = data[data.shape[0] - 3:, data[0][1]:]
    out7 = data[3, data[0][0] - 3:3:data.shape[0] - 3]
    out8 = data[Tensor([[1, 2], [2, 3]]), Tensor([[0, 1], [1, 0]])]

    return out0, out1, out2, out3, out4, out5, out6, out7, out8


@_add_testcase_to_testlist()
def _run_getitem_by_bool(data):
    """test index"""
    out0 = data[Tensor([[True, False, True, False, False],
                        [True, False, True, True, True],
                        [True, False, True, False, True],
                        [False, False, True, True, False],
                       ])]

    out1 = data[True, True]
    return out0, out1

@_add_testcase_to_testlist()
def _run_getitem_by_ellipsis_4d(data):
    """test index"""
    out0 = data[..., 1::2, ::2]
    out1 = data[..., 1::2, ::-2]
    out2 = data[:, ..., ::2]
    out3 = data[:, ..., ::-1]
    out4 = data[::-1, 1::-1, ...]
    return out0, out1, out2, out3, out4

#@_add_testcase_to_testlist()
def _run_new_case(data):
    """test index"""
    out5 = data + 0
    out5[::-1, 1::1, None, ...] = 117 # todo
    return out5

def _run_st(data):
    """test index"""
    out0 = data[::-1]
    out1 = data[None::-2]
    out2 = data[1:None]
    out3 = data[..., ::2, 1::2]
    out4 = data[..., :, None, None]
    out5 = data[..., ::2, 1::2, None]
    out6 = data[::2, ..., 1::2, None]
    out7 = data[::2, ..., 1::2]

    out8 = data + 0
    out8[::2, ..., 1::2] = [111, 112]
    out8[..., ::3, 1::2] = [113, 114]
    out8[..., ::3, 1::2, None] = 113
    return out0, out1, out2, out3, out4, out5, out6, out7, out8

IR_LEVEL = 0
RUN_AND_COMPARE = 0
DEBUG_GRAPH = 1
DEBUG_PYNATIVE = 2
RUN_MODE = RUN_AND_COMPARE

DYNAMIC_LEVEL = 2 # 0: static_shape, 1: dim unknown, 2: rank unknown, 100: (static_shape, dim_unknown and rank_unknown)

def _compare(expect, actual, case_name, index):
    """compare result"""
    if expect.shape != actual.shape:
        print(f"Run {case_name} failed: the shape of {index}'th output is not match to the expect.")
        print(f"The Expect shape is: {expect.shape}")
        print(f"The Actual shape is: {actual.shape}")
        assert False

    res = np.allclose(expect.asnumpy(), actual.asnumpy())
    if not res:
        print(f"Run {case_name} failed: the {index}'th output is not match to the expect.")
        print(f"The Expect is: {expect}")
        print(f"The Actual is: {actual}")
        assert False

def _compare_result(expect_res, actual_res, name):
    """compare result"""
    if isinstance(expect_res, (tuple, list)):
        for index, (expect, actual) in enumerate(zip(expect_res, actual_res)):
            _compare(expect, actual, name, index)
    else:
        _compare(expect_res, actual_res, name, 0)


def _run_and_compare(test_func, data):
    """_run_and_compare"""
    print(f"\n******************************Testing {test_func.__name__}......")
    if RUN_MODE == DEBUG_PYNATIVE:
        pynative_out = test_func(data)
        print("pynative out is:")
        print(pynative_out)
        print(pynative_out.shape)
        return

    if RUN_MODE == DEBUG_GRAPH:
        context.set_context(save_graphs=IR_LEVEL, save_graphs_path='./IR')
        if DYNAMIC_LEVEL == 0:
            graph_out = ms.jit()(test_func)(data)
        elif DYNAMIC_LEVEL == 1:
            dim_unknown_shape = ()
            for _ in data.shape:
                dim_unknown_shape += (None,)
            graph_out = ms.jit(input_signature=(Tensor(shape=dim_unknown_shape, dtype=data.dtype)))(test_func)(data)
        else:
            graph_out = ms.jit(input_signature=(Tensor(shape=None, dtype=data.dtype)))(test_func)(data)
        print("graph out is")
        print(graph_out)
        print(graph_out.shape)
        return

    print("Running with pynative mode...")
    pynative_out = test_func(data)
    print("Running with graph mode...")
    context.set_context(save_graphs=IR_LEVEL, save_graphs_path='./IR')
    dim_unknown_shape = ()
    for _ in data.shape:
        dim_unknown_shape += (None,)
    if DYNAMIC_LEVEL == 0:
        graph_out = ms.jit()(test_func)(data)
    elif DYNAMIC_LEVEL == 1:
        graph_out = ms.jit(input_signature=(Tensor(shape=dim_unknown_shape, dtype=data.dtype)))(test_func)(data)
    elif DYNAMIC_LEVEL == 100:
        print("Static_shape...")
        graph_out = ms.jit()(test_func)(data)
        _compare_result(pynative_out, graph_out, test_func.__name__)

        print("[dynamic_shape with known rank...]")
        graph_out = ms.jit(input_signature=(Tensor(shape=dim_unknown_shape, dtype=data.dtype)))(test_func)(data)
        _compare_result(pynative_out, graph_out, test_func.__name__)

        print("[dynamic_shape with unknown rank...]")
        graph_out = ms.jit(input_signature=(Tensor(shape=None, dtype=data.dtype)))(test_func)(data)
        _compare_result(pynative_out, graph_out, test_func.__name__)
        print(f"Test successfully for {test_func.__name__}.\n")
        return
    else:
        graph_out = ms.jit(input_signature=(Tensor(shape=None, dtype=data.dtype)))(test_func)(data)

    _compare_result(pynative_out, graph_out, test_func.__name__)
    print(f"Test successfully for {test_func.__name__}.\n")


input_data = Tensor(np.arange(4 * 5).reshape((4, 5)))
input_data_4d = Tensor(np.arange(4 * 3 * 4 * 4).reshape((4, 3, 4, 4)))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_index_st():
    """
    Feature: Tensor index
    Description: test setitem and getitem
    Expectation: get correct result
    """
    global DYNAMIC_LEVEL
    DYNAMIC_LEVEL = 100
    _run_and_compare(_run_st, input_data)


class NetWorkFancyIndex(nn.Cell):
    def __init__(self, index):
        super(NetWorkFancyIndex, self).__init__()
        self.index = index

    def construct(self, tensor):
        return tensor[self.index]

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_fancy_index_integer_list_negative(mode):
    """
    Feature: Test fancy index
    Description: Test fancy index with negative index
    Expectation: Success
    """
    context.set_context(mode=mode)
    index = [-3, 0, 2, -1, -1]
    net = NetWorkFancyIndex(index)
    input_np = np.array([1, 2, 3])
    input_me = Tensor(input_np, dtype=ms.float32)
    out_ms = net(input_me)
    assert np.allclose(out_ms.asnumpy(), [1, 1, 3, 3, 3])
    out_grad = Tensor([1, 2, 3, 4, 5], dtype=ms.float32)
    grad_ms = GradOperation(sens_param=True)(net)(input_me, out_grad)
    assert np.allclose(grad_ms.asnumpy(), [3, 0, 12])


class EllipsisIndexNet(nn.Cell):
    def construct(self, x):
        return x[..., 0]

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_index_with_ellipsis(mode):
    """
    Feature: Test index with ellipsis
    Description: Test index with ellipsis
    Expectation: Success
    """
    context.set_context(mode=mode)
    net = EllipsisIndexNet()
    dyn_x = Tensor(shape=[None, 2, 2], dtype=ms.int64)
    net.set_inputs(dyn_x)
    x = ms.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y = net(x)
    assert np.allclose(y.asnumpy(), [[1, 3], [5, 7]])


# usage: python test_index.py RUN_MODE IR_LEVEL
# RUN_MODE: 0-run and compare result, 1-run with graph_mode, 2-run with pynative_mode
# IR_LEVEL: reference to 'save_graphs' in set_context
if __name__ == '__main__':
    if len(sys.argv) > 1:
        RUN_MODE = int(sys.argv[1])
    if len(sys.argv) > 2:
        IR_LEVEL = int(sys.argv[2])

    for func in TEST_LIST:
        data_in = input_data
        if '_4d' in func.__name__:
            data_in = input_data_4d
        _run_and_compare(func, data_in)

# Copyright 2022 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.ops as ops


def init_result(func, shape, dtype):
    result = np.ones(shape, dtype)
    if func == 'sum':
        result = np.zeros(shape, dtype)
    if func == 'min':
        if dtype in [np.int32, np.uint8, np.int16, np.int8, np.int64, np.uint16, np.uint32, np.uint64]:
            result = result * np.iinfo(dtype).max
        if dtype in [np.float32, np.float64]:
            result = result * np.finfo(dtype).max
    if func == 'max':
        if dtype in [np.int32, np.uint8, np.int16, np.int8, np.int64, np.uint16, np.uint32, np.uint64]:
            result = result * np.iinfo(dtype).min
        if dtype in [np.float32, np.float64]:
            result = result * np.finfo(dtype).min
    return result


arith_np_func_map = {
    "prod": lambda a, b: a * b,
    "sum": lambda a, b: a + b,
    "max": np.maximum,
    "min": np.minimum,
}

def unsorted_segment_arith_expected(func, x, segment_ids, num_segments):
    np_inp = x.asnumpy().copy()
    np_ids = segment_ids.asnumpy().copy()

    f = arith_np_func_map.get(func)

    inp_shape = np_inp.shape
    ids_shape = np_ids.shape
    cal_shape = inp_shape[len(ids_shape):]

    out_shape = np.concatenate(([num_segments], cal_shape), axis=0).astype(np.int32)
    result = init_result(func, out_shape, np_inp.dtype)

    inp_size = np_inp.size
    ids_size = np_ids.size
    cal_size = np.int32(result.size / num_segments)

    trans_inp_batch = np.int32(inp_size / cal_size)
    trans_inp_shape = np.concatenate(([trans_inp_batch], cal_shape), axis=0).astype(np.int32)

    trans_inp = np_inp.reshape(trans_inp_shape)
    trans_ids = np_ids.reshape(ids_size)


    for i in range(ids_size):
        out_index = trans_ids[i]
        if out_index < 0:
            continue
        if out_index >= num_segments:
            continue
        result[out_index] = f(result[out_index], trans_inp[i])

    return result


class UnsortedSegmentProdNet(nn.Cell):
    def __init__(self, num_segments):
        super(UnsortedSegmentProdNet, self).__init__()
        self.unsorted_segment_prod = P.UnsortedSegmentProd()
        self.num_segments = num_segments

    def construct(self, data, ids):
        return self.unsorted_segment_prod(data, ids, self.num_segments)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['prod'])
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32, mstype.int32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_1D(func, data_type, index_type):
    """
    Feature: test unsorted segment prod
    Description: Test the function of unsorted segment binary op.
    Expectation: match to numpy benchmark.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_x = Tensor(np.random.randint(0, 100, size=[4]), data_type)
    segment_ids = Tensor(np.random.randint(0, 5, size=[4]), index_type)
    num_segments = 5

    net = UnsortedSegmentProdNet(num_segments)
    output = net(input_x, segment_ids)
    expected = unsorted_segment_arith_expected('prod', input_x, segment_ids, num_segments)
    assert (output.asnumpy() == expected).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['prod'])
@pytest.mark.parametrize('data_type', [mstype.float32, mstype.int32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_2D(func, data_type, index_type):
    """
    Feature: test unsorted segment prod
    Description: Test the function of unsorted segment binary op.
    Expectation: match to numpy benchmark.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_x = Tensor(np.random.randint(0, 100, size=[3, 4]), data_type)
    segment_ids = Tensor(np.random.randint(0, 4, size=[3]), index_type)
    num_segments = 4

    net = UnsortedSegmentProdNet(num_segments)
    output = net(input_x, segment_ids)
    expected = unsorted_segment_arith_expected('prod', input_x, segment_ids, num_segments)
    assert (output.asnumpy() == expected).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['prod'])
@pytest.mark.parametrize('data_type', [mstype.float32, mstype.int32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_3D(func, data_type, index_type):
    """
    Feature: test unsorted segment prod
    Description: Test the function of unsorted segment binary op.
    Expectation: match to numpy benchmark.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    input_x = Tensor(np.random.randint(0, 100, size=[4, 5, 3]), data_type)
    segment_ids = Tensor(np.random.randint(0, 5, size=[4]), index_type)
    num_segments = 5

    net = UnsortedSegmentProdNet(num_segments)
    output = net(input_x, segment_ids)
    expected = unsorted_segment_arith_expected('prod', input_x, segment_ids, num_segments)
    assert (output.asnumpy() == expected).all()


# Testing Dynamic Shape
class UnsortedSegmentProdDynNet(nn.Cell):
    def __init__(self, num_segments, dyn_a=True, dyn_b=True):
        super(UnsortedSegmentProdDynNet, self).__init__()
        self.unsorted_segment_prod = P.UnsortedSegmentProd()
        self.gpu_convert_to_dynamic_shape = inner.GpuConvertToDynamicShape()
        self.num_segments = num_segments
        self.to_dyn_1 = dyn_a
        self.to_dyn_2 = dyn_b
    def construct(self, data, ids):
        # testing selective inputs being dynamic
        if self.to_dyn_1:
            data = self.gpu_convert_to_dynamic_shape(data)
        if self.to_dyn_2:
            ids = self.gpu_convert_to_dynamic_shape(ids)
        return self.unsorted_segment_prod(data, ids, self.num_segments)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['prod'])
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32, mstype.int32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_dyn_ab(func, data_type, index_type):
    """
    Feature: Tests for dynamic shape with both inputs dynamic
    Description: Test the function of unsorted segment binary op.
    Expectation: match to numpy benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    num_segments = 4
    net = UnsortedSegmentProdDynNet(num_segments)

    # test 1
    input_x = Tensor(np.random.randint(0, 100, size=[4]), data_type)
    segment_ids = Tensor(np.random.randint(0, 4, size=[4]), index_type)
    output = net(input_x, segment_ids)
    expected = unsorted_segment_arith_expected('prod', input_x, segment_ids, num_segments)
    assert (output.asnumpy() == expected).all()

    # test 2
    input_x = Tensor(np.random.randint(0, 100, size=[3, 4]), data_type)
    segment_ids = Tensor(np.random.randint(0, 4, size=[3]), index_type)
    output = net(input_x, segment_ids)
    expected = unsorted_segment_arith_expected('prod', input_x, segment_ids, num_segments)
    assert (output.asnumpy() == expected).all()

    # test 3
    input_x = Tensor(np.random.randint(0, 100, size=[4, 5, 3]), data_type)
    segment_ids = Tensor(np.random.randint(0, 4, size=[4]), index_type)
    output = net(input_x, segment_ids)
    expected = unsorted_segment_arith_expected('prod', input_x, segment_ids, num_segments)
    assert (output.asnumpy() == expected).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['prod'])
@pytest.mark.parametrize('data_type', [mstype.float32, mstype.int32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_dyn_a(func, data_type, index_type):
    """
    Feature: Tests for dynamic shape with first input dynamic
    Description: Test the function of unsorted segment binary op.
    Expectation: match to numpy benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    num_segments = 6
    net = UnsortedSegmentProdDynNet(num_segments, True, False)

    # test 1
    input_x = Tensor(np.random.randint(0, 100, size=[4]), data_type)
    segment_ids = Tensor(np.random.randint(0, 4, size=[4]), index_type)
    output = net(input_x, segment_ids)
    expected = unsorted_segment_arith_expected('prod', input_x, segment_ids, num_segments)
    assert (output.asnumpy() == expected).all()

    # test 2
    input_x = Tensor(np.random.randint(0, 100, size=[3, 4]), data_type)
    segment_ids = Tensor(np.random.randint(0, 4, size=[3]), index_type)
    output = net(input_x, segment_ids)
    expected = unsorted_segment_arith_expected('prod', input_x, segment_ids, num_segments)
    assert (output.asnumpy() == expected).all()

    # test 3
    input_x = Tensor(np.random.randint(0, 100, size=[4, 5, 3]), data_type)
    segment_ids = Tensor(np.random.randint(0, 4, size=[4]), index_type)
    output = net(input_x, segment_ids)
    expected = unsorted_segment_arith_expected('prod', input_x, segment_ids, num_segments)
    assert (output.asnumpy() == expected).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['prod'])
@pytest.mark.parametrize('data_type', [mstype.float32, mstype.int32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_dyn_b(func, data_type, index_type):
    """
    Feature: Tests for dynamic shape with second input dynamic
    Description: Test the function of unsorted segment binary op.
    Expectation: match to numpy benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    num_segments = 6
    net = UnsortedSegmentProdDynNet(num_segments, False, True)

    # test 1
    input_x = Tensor(np.random.randint(0, 100, size=[4]), data_type)
    segment_ids = Tensor(np.random.randint(0, 6, size=[4]), index_type)
    output = net(input_x, segment_ids)
    expected = unsorted_segment_arith_expected('prod', input_x, segment_ids, num_segments)
    assert (output.asnumpy() == expected).all()

    # test 2
    input_x = Tensor(np.random.randint(0, 100, size=[3, 4]), data_type)
    segment_ids = Tensor(np.random.randint(0, 6, size=[3]), index_type)
    output = net(input_x, segment_ids)
    expected = unsorted_segment_arith_expected('prod', input_x, segment_ids, num_segments)
    assert (output.asnumpy() == expected).all()

    # test 3
    input_x = Tensor(np.random.randint(0, 100, size=[4, 5, 3]), data_type)
    segment_ids = Tensor(np.random.randint(0, 6, size=[4]), index_type)
    output = net(input_x, segment_ids)
    expected = unsorted_segment_arith_expected('prod', input_x, segment_ids, num_segments)
    assert (output.asnumpy() == expected).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_check():
    """
    Feature: test_tensor_check.
    Description: test cases for tensor func
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = Tensor(np.random.randint(0, 100, size=[4]), mstype.float32)
    segment_ids = Tensor(np.random.randint(0, 5, size=[4]), mstype.int32)
    num_segments = 5

    output_ms = x.unsorted_segment_prod(segment_ids, num_segments)
    expected = unsorted_segment_arith_expected('prod', x, segment_ids, num_segments)
    np.testing.assert_allclose(output_ms.asnumpy(), expected, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_functional_check():
    """
    Feature: test_functional_check.
    Description: test cases for functional func.
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = Tensor(np.random.randint(0, 100, size=[4]), mstype.float32)
    segment_ids = Tensor(np.random.randint(0, 5, size=[4]), mstype.int32)
    num_segments = 5

    output_ms = F.unsorted_segment_prod(x, segment_ids, num_segments)
    expected = unsorted_segment_arith_expected('prod', x, segment_ids, num_segments)
    np.testing.assert_allclose(output_ms.asnumpy(), expected, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vmap():
    """
    Feature: test_vmap.
    Description: in_axes : None, 0, None
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = Tensor(np.random.rand(4, 5).astype(np.float32))
    segment_ids = Tensor(np.random.randint(0, 5, size=[2, 3, 4]), mstype.int32)
    num_segments = 5
    in_axes = (None, 0, None)
    out_axes = 0

    def prod_vmap_graph(x, segment_ids, num_segments):
        return P.UnsortedSegmentProd()(x, segment_ids, num_segments)
    vmap_graph = prod_vmap_graph
    vmap_round_net = ops.vmap(ops.vmap(vmap_graph, in_axes, out_axes), in_axes, out_axes)
    output = vmap_round_net(x, segment_ids, num_segments)

    expected = []
    for i in range(0, 2):
        for j in range(0, 3):
            ids_s = segment_ids[i, j]
            out_s = unsorted_segment_arith_expected('prod', x, ids_s, num_segments)
            expected.append(out_s)

    output_shape = (2, 3, 5, 5)
    expected = np.array(expected).reshape(output_shape)
    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-3)

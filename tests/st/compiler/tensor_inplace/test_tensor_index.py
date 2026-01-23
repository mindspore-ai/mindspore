# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
""" test_tensor_index.py """
import os
import mindspore as ms
from mindspore import nn
from mindspore import context, mint, ops, jit, Tensor
from tests.mark_utils import arg_mark
from tests.st.compiler.utils import match_array

context.set_context(jit_config={"jit_level": "O0"})


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.01)
def test_tensor_select_ext_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[0] = y
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([2], dtype=ms.int32)
        input_y = ms.Tensor([3], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x, input_y)
        print("out:", out)
        assert out == 3
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1",
        card_mark="onecard", essential_mark="unessential", mem_peak=1.00)
def test_tensor_select_ext_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[0]
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([2], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x)
        print("out:", out)
        assert out == 2
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.01)
def test_tensor_slice_ext_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[0: 2] = y
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        input_y = ms.Tensor([3], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x, input_y)
        print("out:", out)
        assert ms.ops.all(out == ms.Tensor([3, 3, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.01)
def test_tensor_slice_ext_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[0:2]
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x)
        print("out:", out)
        assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1",
        card_mark="onecard", essential_mark="unessential", mem_peak=2048.01)
def test_tensor_None_index_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[None] = y
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        input_y = ms.Tensor([3], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x, input_y)
        print("out:", out)
        assert ms.ops.all(out == ms.Tensor([3, 3, 3], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.0)
def test_tensor_None_index_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[None]
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x)
        print("out:", out)
        assert ms.ops.all(out == ms.Tensor([1, 1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1",
        card_mark="onecard", essential_mark="unessential", mem_peak=2048.01)
def test_tensor_self_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[...] = y
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        input_y = ms.Tensor([2], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x, input_y)
        print("out:", out)
        assert ms.ops.all(out == ms.Tensor([2, 2, 2], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1",
        card_mark="onecard", essential_mark="unessential", mem_peak=1024.00)
def test_tensor_self_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[...]
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x)
        print("out:", out)
        assert ms.ops.all(out == ms.Tensor([1, 1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.0)
def test_tensor_bool_index_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y, index):
            x[index] = y
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        input_y = ms.Tensor([2], dtype=ms.int32)
        index = True
        out1 = net(input_x, input_y, index)
        index = False
        out2 = net(input_x, input_y, index)
        print("index_True:", out1)
        print("index_False:", out2)
        assert ms.ops.all(out1 == ms.Tensor([2, 2, 2], dtype=ms.int32))
        assert ms.ops.all(out2 == ms.Tensor([2, 2, 2], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1",
        card_mark="onecard", essential_mark="unessential", mem_peak=2048.01)
def test_tensor_bool_index_read_true():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, index):
            y = x[index]
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        index = True
        out1 = net(input_x, index)
        print("index_True:", out1)
        assert ms.ops.all(out1 == ms.Tensor([1, 1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.01)
def test_tensor_bool_index_read_false():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, index):
            y = x[index]
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        index = 0
        out2 = net(input_x, index)
        print("out2:", out2)
        assert ms.ops.all(out2 == ms.Tensor([], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.01)
def test_tensor_bool_tensor_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y, index):
            x[index] = y
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        input_y = ms.Tensor([2], dtype=ms.int32)
        index = ms.Tensor([True, False, True])
        out = net(input_x, input_y, index)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([2, 1, 2], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1",
        card_mark="onecard", essential_mark="unessential", mem_peak=2048.01)
def test_tensor_bool_tensor_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, index):
            y = x[index]
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        index = ms.Tensor([True, False, True])
        out = net(input_x, index)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.0)
def test_tensor_select_slice_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[0, 0:2] = y
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([[1, 1, 1]], dtype=ms.int32)
        y = ms.Tensor([2], dtype=ms.int32)
        out = net(x, y)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([[2, 2, 1]], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1",
        card_mark="onecard", essential_mark="unessential", mem_peak=2048.01)
def test_tensor_select_slice_write_2():
    """
    Feature: Support tensor inplace.
    Description: Tensor setitem by slice, then use this tensor to do some op.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x: ms.Tensor):
            x[1:] = 1
            return x + 1

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([[1, 2], [3, 4]])
        out = net(x)
        assert ms.ops.all(out == ms.Tensor([[2, 3], [2, 2]]))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.01)
def test_tensor_select_slice_write_3():
    """
    Feature: Support tensor inplace.
    Description: Tensor setitem by slice, then use this tensor to do some op.
    Expectation: Run success; CSE pass shouldn't eliminate the second zeros_like node.
    """

    class Net(nn.Cell):
        def construct(self, x: ms.Tensor):
            a = mint.zeros_like(x)  # AbstractRefTensor-1
            a[1:] = 1
            b = a + 1
            a = mint.zeros_like(x)  # AbstractRefTensor-2, shouldn't be eliminated by CSE pass.
            a[0:1] = 2
            c = a * 2
            return b + c

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ops.randn(2, 2)
        result = net(x)
        expect = ms.Tensor([[5., 5.], [2., 2.]], dtype=ms.float32)
        assert ops.all(expect == result)
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.0)
def test_tensor_select_slice_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[0, 0:2]
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([[1, 1, 1]], dtype=ms.int32)
        out = net(x)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.0)
def test_tensor_index_list_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[[0, 2]] = y
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        y = ms.Tensor([2], dtype=ms.int32)
        out = net(x, y)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([2, 1, 2], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.01)
def test_tensor_index_list_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[0: 2]
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        out = net(x)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.0)
def test_tensor_index_tensor_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y, index):
            x[index] = y
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        y = ms.Tensor([2], dtype=ms.int32)
        index = ms.Tensor([0, 2], dtype=ms.int32)
        out = net(x, y, index)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([2, 1, 2], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1",
        card_mark="onecard", essential_mark="unessential", mem_peak=2048.01)
def test_tensor_index_tensor_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, index):
            y = x[index]
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        index = ms.Tensor([0, 2], dtype=ms.int32)
        out = net(x, index)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1",
        card_mark="onecard", essential_mark="unessential", mem_peak=2048.00)
def test_tensor_select_slice_index_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[0, 0:1, [0, 2]] = y
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([[[1, 1, 1]]], dtype=ms.int32)
        y = ms.Tensor([2], dtype=ms.int32)
        out = net(x, y)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([[[2, 1, 2]]], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
        essential_mark="unessential", mem_peak=2048.0)
def test_tensor_select_slice_index_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[0, 0:1, [0, 2]]
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([[[1, 1, 1]]], dtype=ms.int32)
        out = net(x)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([[1, 1]], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_getitem_setitem_by_slice_v1():
    """
    Feature: Test Tensor getitem/setitem.
    Description: Test two tensors getitem by slice, and the slice is a variable.
    Expectation: no exception, no graph break.
    """

    def fn(x, kv_cache, pe_cache, start: int):
        B = x.shape[0]
        y = x + 1
        # start is mutable, so the slice is variable, not constant.
        kv_cache[:B, start : start + 1] = y
        pe_cache[:B, start : start + 1] = y + 1
        return kv_cache + pe_cache

    x = ops.randn(2, 1, 2, dtype=ms.float32)
    kv_cache1 = ops.zeros((2, 2, 2), dtype=ms.float32)
    pe_cache1 = ops.zeros((2, 2, 2), dtype=ms.float32)
    start = ms.mutable(1)  # Use mutable to ensure the slice is variable

    o1 = fn(x, kv_cache1, pe_cache1, start)

    compiled_fn = jit(fn, capture_mode='ast', fullgraph=True)
    kv_cache2 = ops.zeros((2, 2, 2), dtype=ms.float32)
    pe_cache2 = ops.zeros((2, 2, 2), dtype=ms.float32)
    o2 = compiled_fn(x, kv_cache2, pe_cache2, start)

    match_array(o1, o2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_getitem_setitem_by_slice_v2():
    """
    Feature: Test Tensor getitem/setitem.
    Description: Test two tensors getitem by slice, and the slice is a variable.
    Expectation: no exception, no graph break.
    """

    def fn(x, kv_cache, pe_cache, start: int):
        y = x + 1
        # start is mutable, so the slice is variable, not constant.
        kv_cache[start : start + 1] = y
        pe_cache[start : start + 1] = y + 1
        return kv_cache + pe_cache

    x = ops.randn(1, 2, dtype=ms.float32)
    kv_cache1 = ops.zeros((2, 2), dtype=ms.float32)
    pe_cache1 = ops.zeros((2, 2), dtype=ms.float32)
    start = ms.mutable(1)  # Use mutable to ensure the slice is variable

    o1 = fn(x, kv_cache1, pe_cache1, start)

    compiled_fn = jit(fn, capture_mode='ast', fullgraph=True)
    kv_cache2 = ops.zeros((2, 2), dtype=ms.float32)
    pe_cache2 = ops.zeros((2, 2), dtype=ms.float32)
    o2 = compiled_fn(x, kv_cache2, pe_cache2, start)

    match_array(o1, o2)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard",
        essential_mark="essential", mem_peak=4.00)
def test_Depend_infer_of_tuple_ref_tensors():
    """
    Feature: Test infer of Tensor setitem.
    Description: Test the infer logic of Depend node, and the first input of
    Depend node is a tuple of inplace modified tensors.
    Expectation: Run success.
    """

    def fn(x: Tensor, y: Tensor, a: Tensor, b: Tensor):
        x2, y2 = f2(x, y, a, b)
        return x2 + y2

    def f2(x: Tensor, y: Tensor, a: Tensor, b: Tensor):
        # Three key conditions to reproduce this issue:
        # 1.The return value is a tuple of ref tensors.
        # 2.The graph contains a side-effect op (e.g., Tensor.copy_()). This op adds Depend nodes to the return value
        # (the aforementioned tuple of ref tensors) to ensure the side-effect op is not eliminated.
        # 3.The side-effect op's return value must be a variable, not a constant (e.g., None).
        # It will trigger a bug in the infer logic of Depend node. It loses the inplace_abstract for elements in the
        # tuple, making the ref tensors in the tuple inferred as non-ref tensors.
        x[0] = a  # inplace modification
        y[0] = b  # inplace modification
        a.copy_(b)  # side-effect op, will insert a Depend node
        return x, y  # return a tuple of ref-tensors

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

        x1 = ops.randn(2, 2)
        y1 = ops.randn(2, 2)
        a1 = ops.randn(2)
        b1 = ops.randn(2)

        x2 = x1.copy()
        y2 = y1.copy()
        a2 = a1.copy()
        b2 = b1.copy()

        o1 = fn(x1, y1, a1, b1)

        compiled_fn = jit(fn, fullgraph=True, backend="ms_backend")
        o2 = compiled_fn(x2, y2, a2, b2)

        match_array(o1, o2)
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]

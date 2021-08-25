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

import inspect
import numpy as np
import pytest
from mindspore import context, ops, Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Cell


class UserDefined(ops.PrimitiveWithInfer):
    def __init__(self, func, shape, dtype, func_type=None):
        ops.PrimitiveWithInfer.__init__(self, "UserDefined")
        self.add_prim_attr('akg', True)

        if "__wrapped__" in func.__dict__:
            func = func.__dict__["__wrapped__"]
        func_name = func.__name__
        self.add_prim_attr('func_name', func_name)
        func_source_str = inspect.getsource(func)

        if func_type is None:
            if "ir_builder" in func_source_str:
                func_type = "ir_builder"
            elif "compute" in func_source_str:
                func_type = "tvm_compute"
            else:
                func_type = "hybrid"

        self.add_prim_attr('func_source_str', func_source_str)
        self.add_prim_attr('func_type', func_type)

        self._shape = shape
        self._dtype = dtype

    def infer_shape(self, *args):
        if callable(self._shape):
            return self._shape(*args)
        return self._shape

    def infer_dtype(self, *args):
        if callable(self._dtype):
            return self._dtype(*args)
        return self._dtype


def outer_product(a, b):
    c = output_tensor((a.shape[0], b.shape[1]), 'float32')

    for i0 in range(a.shape[0]):
        for i1 in range(b.shape[1]):
            c[i0, i1] = 0.0
            for i2 in range(a.shape[1]):
                c[i0, i1] = c[i0, i1] + (a[i0, i2] * b[i2, i1])
    return c


class TestHybrid(Cell):
    def __init__(self):
        super(TestHybrid, self).__init__()

        def infer_func(x, y):
            return x

        self.program = UserDefined(
            outer_product, shape=infer_func, dtype=infer_func)

    def construct(self, x, y):
        return self.program(x, y)


def v_add(inputs, attrs):
    def vadd_func(dst, data_1, data_2):
        ib = tvm.ir_builder.create()
        with ib.for_range_n(data_1.shape, "i") as i:
            ib.store(dst, i, ib.load(data_1, i) + ib.load(data_2, i))
        return ib.get()
    data_1, data_2 = inputs[0], inputs[1]
    return tvm.extern(data_1.shape, [data_1, data_2],
                      lambda ins, outs: vadd_func(outs[0], ins[0], ins[1]),
                      name="v_add", dtype=data_1.dtype)


class TestIRbuilder(Cell):
    def __init__(self, shape):
        super(TestIRbuilder, self).__init__()
        self.program = UserDefined(
            v_add, shape=shape, dtype=mstype.float16)

    def construct(self, x, y):
        return self.program(x, y)


def test_user_defined_hybrid():

    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float32)
    input_y = np.random.normal(0, 1, [4, 4]).astype(np.float32)

    test = TestHybrid()
    output = test(Tensor(input_x), Tensor(input_y))
    expect = np.matmul(input_x, input_y)
    assert np.allclose(expect, output.asnumpy(), 0.001, 0.001)


def test_user_defined_irbuider():

    shape = (4, 5)
    input_x = np.random.normal(0, 1, shape).astype(np.float16)
    input_y = np.random.normal(0, 1, shape).astype(np.float16)

    test = TestIRbuilder(shape)
    output = test(Tensor(input_x), Tensor(input_y))
    assert np.allclose(input_x + input_y, output.asnumpy(), 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_user_defined_gpu():
    context.set_context(mode=0, enable_graph_kernel=True)
    test_user_defined_hybrid()
    test_user_defined_irbuider()

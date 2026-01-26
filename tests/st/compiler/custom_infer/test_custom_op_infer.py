# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Test custom op infer"""

import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, jit
from mindspore.common import dtype as mstype
from mindspore.graph import register_op_infer, register_op_wrapper, clear_op_infer, clear_op_wrapper
from tests.mark_utils import arg_mark


def teardown_module():
    clear_op_infer()
    clear_op_wrapper()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_op_infer_1():
    """
    Feature: Custom op infer.
    Description: Test register_op_infer decorator function.
    Expectation: Custom infer set to the result of operator successfully.
    """
    class InferNet(ms.nn.Cell):
        @jit(backend='ms_backend')
        def construct(self, x, y, z):
            x = ops.Add()(x, z)
            x = ops.Sub()(x, y)
            return x

    @register_op_infer("Add", 'layout')
    def add_infer_layout(x, b):
        return x.shape + b.shape

    @register_op_infer("Sub", 'layout')
    def sub_infer_layout(x, b):
        assert x.layout == (2, 3, 2, 3)
        return x.shape

    net = InferNet()
    input_x = Tensor([[1., 2., 3.], [4., 5., 6.]])
    input_y = Tensor([[1., 2., 3.], [4., 5., 6.]])
    net(input_x, input_y, input_x)
    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_op_infer_2():
    """
    Feature: Custom op infer.
    Description: Test register_op_infer decorator function.
    Expectation: Custom infer set to the result of operator successfully.
    """
    class InferNet(ms.nn.Cell):
        @jit(backend='ms_backend')
        def construct(self, x, y, z):
            x = ops.Add()(x, y)
            x = ops.Sub()(x, z)
            return x

    @register_op_infer("Add", 'shape')
    def infer_shape(x, b):
        return [x.shape[i] + b.shape[i] for i in range(len(x.shape))]

    @register_op_infer("Add", 'dtype')
    def infer_dtype(x, b):
        return b.dtype

    @register_op_infer("Add", 'layout')
    def add_infer_layout(x, b):
        return x.shape + b.shape

    @register_op_infer("Sub", 'layout')
    def subd_infer_layout(x, b):
        assert x.layout == (2, 3, 2, 3)
        assert x.dtype == mstype.float64
        assert x.shape == (4, 6)
        return x.shape

    net = InferNet()
    input_x = Tensor([[1., 2., 3.], [4., 5., 6.]], dtype=mstype.float32)
    input_y = Tensor([[1, 2, 3], [4, 5, 6]], dtype=mstype.float64)
    net(input_x, input_y, input_x)
    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_op_infer_with_wrapper():
    """
    Feature: Custom op infer and Custom op wrapper.
    Description: Test register_op_infer abd register_op_wrapper decorator function.
    Expectation: Custom infer set to the result of operator and influence the op wrapper successfully.
    """
    class InferNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul = ops.MatMul()

        @jit(backend='ms_backend')
        def construct(self, x, y, z):
            x = self.matmul(x, y)
            x = x.reshape(3, 2)
            x = ops.Add()(x, z)
            return x

    @register_op_infer("Add", 'layout')
    def add_infer_layout(x, b):
        return x.shape

    @register_op_infer("MatMul", 'layout')
    def matmul_infer_layout(x, b, b1, b2):
        return x.shape

    @register_op_wrapper("Reshape")
    def newreshape(x, b):
        layout = x.layout

        def func(x, b):
            return x.reshape(layout)
        return func

    @register_op_infer("Reshape", 'layout')
    def infer_shape(x, b):
        return x.shape

    @register_op_wrapper("Add")
    def newadd(x, b):
        if x.shape == (2, 3):
            def func(x, b):
                return ops.Add()(x, x)
        else:
            def func(x, b):
                return ops.Add()(x, b)
        return func

    net = InferNet()
    input_x = Tensor([[1., 2., 3.], [4., 5., 6.]])
    input_y = Tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    res = net(input_x, input_y, input_x)
    assert res.shape == (2, 3)
    assert np.allclose(res.asnumpy(), np.array(
        [[60., 72., 84.], [132., 162., 192.]]))
    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_op_infer_attribute_propagation():
    """
    Feature: Custom op infer attribute propagation.
    Description: Test that attributes set by register_op_infer propagate through the computation graph.
    Expectation: Attributes are correctly propagated to subsequent operations.
    """
    class PropagationNet(ms.nn.Cell):
        @jit(backend='ms_backend')
        def construct(self, x, y):
            z = ops.Add()(x, y)
            result = ops.Mul()(z, x)
            return result

    @register_op_infer("Add", 'custom_attr')
    def infer_custom_attr_add(x, y):
        return {"operation": "add", "input_shapes": [x.shape, y.shape]}

    @register_op_infer("Mul", 'custom_attr')
    def infer_custom_attr_mul(x, y):
        assert hasattr(
            x, 'custom_attr'), "x should have custom_attr from Add operation"
        assert x.custom_attr[
            "operation"] == "add", f"Expected operation 'add', got {x.custom_attr['operation']}"
        assert x.custom_attr["input_shapes"] == [
            x.shape, y.shape], f"Input shapes mismatch: {x.custom_attr['input_shapes']} vs {[x.shape, y.shape]}"
        return {"operation": "mul", "prev_operation": x.custom_attr["operation"]}

    net = PropagationNet()
    x = Tensor([[1., 2.], [3., 4.]])
    y = Tensor([[1., 1.], [1., 1.]])
    net(x, y)
    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_op_infer_multiple_attributes():
    """
    Feature: Custom op infer multiple attributes.
    Description: Test that multiple attributes can be registered for the same operation.
    Expectation: All attributes are correctly set and accessible.
    """
    class MultiAttrNet(ms.nn.Cell):
        @jit(backend='ms_backend')
        def construct(self, x, y):
            z = ops.Add()(x, y)
            result = ops.Sub()(z, x)
            return result

    @register_op_infer("Add", 'attr1')
    def infer_attr1(x, y):
        return "attribute_1_value"

    @register_op_infer("Add", 'attr2')
    def infer_attr2(x, y):
        return [x.shape[0], y.shape[1]]

    @register_op_infer("Add", 'attr3')
    def infer_attr3(x, y):
        return {"x_dtype": str(x.dtype), "y_dtype": str(y.dtype)}

    @register_op_infer("Sub", 'verification')
    def verify_multiple_attrs(x, y):
        assert hasattr(x, 'attr1'), "x should have attr1 from Add operation"
        assert x.attr1 == "attribute_1_value", f"attr1 mismatch: expected 'attribute_1_value', got {x.attr1}"

        assert hasattr(x, 'attr2'), "x should have attr2 from Add operation"
        expected_attr2 = [x.shape[0], y.shape[1]]
        assert x.attr2 == expected_attr2, f"attr2 mismatch: expected {expected_attr2}, got {x.attr2}"

        assert hasattr(x, 'attr3'), "x should have attr3 from Add operation"
        expected_attr3 = {"x_dtype": str(x.dtype), "y_dtype": str(y.dtype)}
        assert x.attr3 == expected_attr3, f"attr3 mismatch: expected {expected_attr3}, got {x.attr3}"

        return "attributes_verified"

    net = MultiAttrNet()
    x = Tensor([[1., 2.], [3., 4.]])
    y = Tensor([[1., 1.], [1., 1.]])
    net(x, y)
    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_op_infer_complex_chain():
    """
    Feature: Custom op infer complex computation chain.
    Description: Test attribute propagation through a complex chain of operations.
    Expectation: Attributes are correctly propagated through the entire chain.
    """
    class ComplexChainNet(ms.nn.Cell):
        @jit(backend='ms_backend')
        def construct(self, x, y, z):
            a = ops.Add()(x, y)
            b = ops.Mul()(a, z)
            c = ops.Sub()(b, x)
            d = ops.Div()(c, y)
            return d

    @register_op_infer("Add", 'computation_path')
    def infer_path_add(x, y):
        return ["Add"]

    @register_op_infer("Mul", 'computation_path')
    def verify_mul_path(x, y):
        assert hasattr(
            x, 'computation_path'), "x should have computation_path from Add"
        assert x.computation_path == [
            "Add"], f"Expected path ['Add'], got {x.computation_path}"
        path = x.computation_path.copy()
        path.append("Mul")
        return path

    @register_op_infer("Sub", 'computation_path')
    def verify_sub_path(x, y):
        assert hasattr(
            x, 'computation_path'), "x should have computation_path from Mul"
        assert x.computation_path == [
            "Add", "Mul"], f"Expected path ['Add', 'Mul'], got {x.computation_path}"
        path = x.computation_path.copy()
        path.append("Sub")
        return path

    @register_op_infer("Div", 'computation_path')
    def verify_div_path(x, y):
        assert hasattr(
            x, 'computation_path'), "x should have computation_path from Sub"
        assert x.computation_path == [
            "Add", "Mul", "Sub"], f"Expected path ['Add', 'Mul', 'Sub'], got {x.computation_path}"
        path = x.computation_path.copy()
        path.append("Div")
        return path

    net = ComplexChainNet()
    x = Tensor([[4., 8.], [12., 16.]])
    y = Tensor([[2., 2.], [3., 4.]])
    z = Tensor([[1., 2.], [2., 1.]])
    net(x, y, z)
    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_op_infer_error_invalid_op_name():
    """
    Feature: Custom op infer error handling.
    Description: Test error handling for invalid operator names.
    Expectation: Appropriate error is raised for invalid operator names.
    """
    with pytest.raises(ValueError, match="op_name must be a non-empty string"):
        @register_op_infer("", 'test_attr')
        def dummy_func(x, y):
            return "test"

    with pytest.raises(ValueError, match="op_name must be a non-empty string"):
        @register_op_infer(None, 'test_attr')
        def dummy_func(x, y):
            return "test"

    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_op_infer_error_invalid_attr_name():
    """
    Feature: Custom op infer error handling.
    Description: Test error handling for invalid attribute names.
    Expectation: Appropriate error is raised for invalid attribute names.
    """
    with pytest.raises(ValueError, match="infer_name must be a non-empty string"):
        @register_op_infer("Add", '')
        def dummy_func(x, y):
            return "test"

    with pytest.raises(ValueError, match="infer_name must be a non-empty string"):
        @register_op_infer("Add", None)
        def dummy_func(x, y):
            return "test"

    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_op_infer_error_duplicate_registration():
    """
    Feature: Custom op infer error handling for duplicate registration.
    Description: Test error handling when registering the same attribute twice for the same operator.
    Expectation: Appropriate error is raised for duplicate attribute registration.
    """
    @register_op_infer("Add", 'duplicate_attr')
    def infer_duplicate_1(x, y):
        return "first_value"

    with pytest.raises(RuntimeError, match="multiple times"):
        @register_op_infer("Add", 'duplicate_attr')
        def infer_duplicate_2(x, y):
            return "second_value"

    teardown_module()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_op_infer_nonexistent_operator():
    """
    Feature: Custom op infer for nonexistent operator.
    Description: Test registering custom inference for an operator that doesn't exist.
    Expectation: Registration should succeed but have no effect.
    """
    class NonExistentOpNet(ms.nn.Cell):
        @jit(backend='ms_backend')
        def construct(self, x, y):
            return ops.Add()(x, y)

    @register_op_infer("NonExistentOperator", 'test_attr')
    def infer_nonexistent(x, y):
        raise RuntimeError("should_not_be_called")

    net = NonExistentOpNet()
    x = Tensor([[1., 2.], [3., 4.]])
    y = Tensor([[1., 1.], [1., 1.]])
    net(x, y)
    teardown_module()

# Copyright 2025 Huawei Technologies Co., Ltd
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
""" test_tensor_scatter_add.py """
import numpy as np
import pytest
import tensorflow as tf

import mindspore as ms
from mindspore import ops, context, Tensor
from mindspore.common import dtype as mstype
from mindspore.graph.api import _pynative_executor
from mindspore.nn import Cell

from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs, allclose_nparray

tf.compat.v1.disable_eager_execution()

class TensorScatterAddNet(Cell):
    def __init__(self):
        super().__init__()
        self.tensorscatteradd = ops.tensor_scatter_add

    def construct(self, input_x, indices, updates):
        return self.tensorscatteradd(input_x, indices, updates)


class TestModule():
    def __init__(self, inputs=None):
        self.ms_dtype = inputs[0].dtype
        self.input_x = inputs[0]
        self.input_x_np = inputs[0].asnumpy()

        self.indices = inputs[1]
        self.indices_np = inputs[1].asnumpy()

        self.updates = inputs[2]
        self.updates_np = inputs[2].asnumpy()

        if self.ms_dtype == mstype.float16:
            self.loss = 1e-3
        elif self.ms_dtype == mstype.float32:
            self.loss = 1e-4
        elif self.ms_dtype == mstype.bfloat16:
            self.loss = 4e-3
        else:
            self.loss = 0
        self.supports_grad = self.ms_dtype in (mstype.float16, mstype.float32, mstype.bfloat16)

    def forward_mindspore_impl(self):
        net = TensorScatterAddNet()
        out = net(self.input_x, self.indices, self.updates)
        return out.asnumpy()

    def grad_mindspore_impl(self):
        net = TensorScatterAddNet()
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        input_x_grad, _, updates_grad = grad_net(self.input_x, self.indices, self.updates,
                                                 Tensor(self.forward_mindspore_impl()))
        return input_x_grad.asnumpy(), updates_grad.asnumpy()

    def forward_tensorflow_impl(self):
        # Choose appropriate compute precision by dtype
        if self.ms_dtype in (mstype.float16, mstype.float32, mstype.bfloat16):
            # For float types, compute in float32 to improve numerical stability
            input_x = tf.Variable(self.input_x_np.astype(np.float32))
            updates = tf.Variable(self.updates_np.astype(np.float32))
        else:
            # Keep original dtype for integer types
            input_x = tf.Variable(self.input_x_np)
            updates = tf.Variable(self.updates_np)

        indices = tf.Variable(self.indices_np)
        op_tf = tf.raw_ops.TensorScatterAdd(tensor=input_x, indices=indices, updates=updates)
        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            out = sess.run(op_tf)

        # Return result cast to the same dtype as MindSpore input
        if self.ms_dtype in (mstype.float16, mstype.float32, mstype.bfloat16):
            np_dtype = np.float16 if self.ms_dtype == mstype.float16 else np.float32
            return out.astype(np_dtype)

        # For integer types, return as-is without casting
        return out

    def grad_tensorflow_impl(self):
        input_x = tf.Variable(self.input_x_np.copy().astype(np.float32), trainable=True)
        updates = tf.Variable(self.updates_np.copy().astype(np.float32), trainable=True)
        indices = tf.Variable(self.indices_np.copy(), trainable=True)
        net = tf.raw_ops.TensorScatterAdd(tensor=input_x, indices=indices, updates=updates)
        dx = tf.gradients(ys=net, xs=[input_x, updates],
                          grad_ys=self.forward_mindspore_impl().astype(np.float32))
        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            out_tf = sess.run(dx)
        # Return gradients cast to the same dtype as the MindSpore input
        np_dtype = np.float16 if self.ms_dtype == mstype.float16 else np.float32
        return out_tf[0].astype(np_dtype), out_tf[1].astype(np_dtype)

    def forward_cmp(self):
        out_me = self.forward_mindspore_impl()
        out_tf = self.forward_tensorflow_impl()
        # Handle bfloat16 compatibility issue by converting to float32 for comparison
        if str(out_me.dtype) == 'bfloat16':
            out_me = out_me.astype(np.float32)
            out_tf = out_tf.astype(np.float32)
        allclose_nparray(out_tf, out_me, self.loss, self.loss)

    def grad_cmp(self):
        if not self.supports_grad:
            return
        out_me = self.grad_mindspore_impl()
        out_tf = self.grad_tensorflow_impl()
        # Handle bfloat16 compatibility issue by converting to float32 for comparison
        if str(out_me[0].dtype) == 'bfloat16':
            out_me = (out_me[0].astype(np.float32), out_me[1].astype(np.float32))
            out_tf = (out_tf[0].astype(np.float32), out_tf[1].astype(np.float32))
        allclose_nparray(out_tf[0], out_me[0], self.loss, self.loss)
        allclose_nparray(out_tf[1], out_me[1], self.loss, self.loss)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.00)
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32, mstype.bfloat16])
@pytest.mark.parametrize('index_type', [mstype.int32, mstype.int64])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_different_type(data_type, index_type, context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with different data types
    Expectation: the result match between MindSpore and TensorFlow.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 0], [1, 1]]), index_type)
    updates = Tensor(np.array([1.0, 2.2]), data_type)
    module = TestModule(inputs=[input_x, indices, updates])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('data_type', [mstype.int32, mstype.int8, mstype.uint8])
@pytest.mark.parametrize('index_type', [mstype.int32, mstype.int64])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_integer_types_forward_only(data_type, index_type, context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: For integer and unsigned integer data types, only forward comparison (no gradients).
    Expectation: MindSpore forward matches TensorFlow forward.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), data_type)
    indices = Tensor(np.array([[0, 1], [1, 2]]), index_type)
    updates = Tensor(np.array([10, -2]), data_type)
    module = TestModule(inputs=[input_x, indices, updates])
    module.forward_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential',
        mem_peak=0.01)
@pytest.mark.parametrize('shape', [
    (3,),
    (2, 3),
    (3, 4, 5),
    (2, 3, 4, 5),
    (2, 2, 3, 4, 5),
    (2, 2, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2, 2),
    (2, 2, 2, 2, 2, 2, 2, 2),
])
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_different_dimensions(shape, context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with different dimensions (1D-8D)
    Expectation: the result match between MindSpore and TensorFlow.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.random.randn(*shape), mstype.float32)
    indices = Tensor(np.random.randint(0, 2, (3, 2, 1)), mstype.int32)
    updates_shape = indices.shape[:-1] + input_x.shape[indices.shape[-1]:]
    updates = Tensor(np.random.randn(*updates_shape), mstype.float32)
    module = TestModule(inputs=[input_x, indices, updates])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_empty_tensor(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with empty tensor
    Expectation: the result match between MindSpore and TensorFlow.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([]).reshape(0, 3), mstype.float32)
    indices = Tensor(np.array([]).reshape(0, 2), mstype.int32)
    updates = Tensor(np.array([]), mstype.float32)
    module = TestModule(inputs=[input_x, indices, updates])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_inf_nan(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with inf and nan values
    Expectation: the result match between MindSpore and TensorFlow.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([[np.inf, np.nan, 3.6], [0.4, -np.inf, -3.2]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.array([1.0, 2.2]), mstype.float32)
    module = TestModule(inputs=[input_x, indices, updates])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_large_values(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with large values
    Expectation: the result match between MindSpore and TensorFlow.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([[1e10, -1e10, 0], [1e-10, -1e-10, 1e5]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.array([1e8, -1e8]), mstype.float32)
    module = TestModule(inputs=[input_x, indices, updates])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_single_element(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with single element tensor
    Expectation: the result match between MindSpore and TensorFlow.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([[5.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 0]]), mstype.int32)
    updates = Tensor(np.array([2.0]), mstype.float32)
    module = TestModule(inputs=[input_x, indices, updates])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_duplicate_indices(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with duplicate indices
    Expectation: the result match between MindSpore and TensorFlow.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.array([1.0, 2.0, 3.0]), mstype.float32)
    module = TestModule(inputs=[input_x, indices, updates])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_zero_updates(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with zero updates
    Expectation: the result match between MindSpore and TensorFlow.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.array([0.0, 0.0]), mstype.float32)
    module = TestModule(inputs=[input_x, indices, updates])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_implicit_cast_not_supported_raises(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with implicit type conversion.
    Expectation: Both MindSpore and TensorFlow raise exception.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.random.randn(3, 4).astype(np.float32), mstype.float32)
    indices = Tensor(np.array([[0], [1], [2]]), mstype.int32)
    # Error: updates dtype (float16) doesn't match input_x dtype (float32)
    updates = Tensor(np.random.randn(3, 4).astype(np.float16), mstype.float16)

    module = TestModule(inputs=[input_x, indices, updates])
    # Test MindSpore raises TypeError
    with pytest.raises(TypeError):
        module.forward_mindspore_impl()
        _pynative_executor.sync()

    # Test TensorFlow raises TypeError (direct call without type conversion)
    input_x_np = input_x.asnumpy()
    indices_np = indices.asnumpy()
    updates_np = updates.asnumpy()
    tf_input_x = tf.Variable(input_x_np)
    tf_indices = tf.Variable(indices_np)
    tf_updates = tf.Variable(updates_np)  # Keep original float16 dtype
    with pytest.raises(TypeError):
        tf.raw_ops.TensorScatterAdd(tensor=tf_input_x, indices=tf_indices, updates=tf_updates)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential',
        mem_peak=2048.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_input_value_range_float(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with random float value ranges.
    Expectation: MindSpore and TensorFlow forward results are consistent; float types also verify gradients.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    # Generate random input with various value ranges
    input_x = Tensor(np.random.randn(10, 4).astype(np.float32), mstype.float32)
    indices = Tensor(np.random.randint(0, 10, (6, 1)).astype(np.int32), mstype.int32)
    updates = Tensor(np.random.randn(6, 4).astype(np.float32), mstype.float32)
    module = TestModule(inputs=[input_x, indices, updates])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_k_less_than_rank(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: K (last dim of indices) < rank(input); slice updates on higher-rank tensors.
    Expectation: MindSpore and TensorFlow results are consistent.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    # Example shape (2,3,4), K=2: index first two dims, add along the last dim
    input_x = Tensor(np.random.randn(2, 3, 4).astype(np.float32), mstype.float32)
    indices = Tensor(np.array([[0, 1], [1, 2]]).astype(np.int32), mstype.int32)
    updates = Tensor(np.random.randn(2, 4).astype(np.float32), mstype.float32)
    module = TestModule(inputs=[input_x, indices, updates])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_k_equal_rank_multi_updates(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: K equals input rank; multiple random updates, including repeated indices.
    Expectation: MindSpore and TensorFlow results are consistent.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    shape = (5, 6)
    input_x = Tensor(np.random.randn(*shape).astype(np.float32), mstype.float32)
    num_updates = 20
    indices_np = np.column_stack([np.random.randint(0, d, (num_updates,)) for d in shape]).astype(np.int32)
    indices = Tensor(indices_np, mstype.int32)
    updates = Tensor(np.random.randn(num_updates).astype(np.float32), mstype.float32)
    module = TestModule(inputs=[input_x, indices, updates])
    module.forward_cmp()
    module.grad_cmp()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_updates_shape_not_match_with_input(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with shape mismatch should raise exception.
    Expectation: MindSpore raises exception.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.zeros((3, 3), dtype=np.float32), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]], dtype=np.int32), mstype.int32)
    updates = Tensor(np.array([1.0], dtype=np.float32), mstype.float32)
    net = TensorScatterAddNet()
    with pytest.raises(ValueError):
        net(input_x, indices, updates)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_dtype_mismatch_raises(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with dtype mismatch should raise exception.
    Expectation: MindSpore raises exception.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.zeros((3, 3), dtype=np.float32), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]], dtype=np.int32), mstype.int32)
    # Error: updates dtype doesn't match input_x dtype
    updates = Tensor(np.array([1.0, 2.0], dtype=np.float64), mstype.float64)
    net = TensorScatterAddNet()
    with pytest.raises(TypeError):
        net(input_x, indices, updates)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_indices_rank_exceed_raises(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with indices.shape[-1] > rank(input_x) should raise exception.
    Expectation: MindSpore raises exception.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    # input_x is 2D tensor (rank=2)
    input_x = Tensor(np.zeros((3, 3), dtype=np.float32), mstype.float32)
    # Error: indices.shape[-1] = 3 > rank(input_x) = 2
    indices = Tensor(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32), mstype.int32)
    updates = Tensor(np.array([1.0, 2.0], dtype=np.float32), mstype.float32)
    net = TensorScatterAddNet()
    with pytest.raises(ValueError):
        net(input_x, indices, updates)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_indices_rank_less_than_2_raises(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with rank(indices) < 2 should raise exception.
    Expectation: MindSpore raises exception.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    input_x = Tensor(np.zeros((3, 3), dtype=np.float32), mstype.float32)
    # Error: indices is 1D tensor (rank=1), but should be at least 2D
    indices = Tensor(np.array([0, 0], dtype=np.int32), mstype.int32)
    updates = Tensor(np.array([1.0], dtype=np.float32), mstype.float32)
    net = TensorScatterAddNet()
    with pytest.raises(ValueError):
        net(input_x, indices, updates)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_scatter_add_input_not_tensor_raises(context_mode):
    """
    Feature: TensorScatterAdd operators.
    Description: test cases for TensorScatterAdd operator with input_x not being Tensor type should raise exception.
    Expectation: MindSpore raises exception.
    """
    context.set_context(mode=context_mode, jit_level='O0', device_target='Ascend')
    # Error: input_x is numpy array, not Tensor
    input_x = np.zeros((3, 3), dtype=np.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]], dtype=np.int32), mstype.int32)
    updates = Tensor(np.array([1.0, 2.0], dtype=np.float32), mstype.float32)
    net = TensorScatterAddNet()
    with pytest.raises(TypeError):
        net(input_x, indices, updates)
        _pynative_executor.sync()

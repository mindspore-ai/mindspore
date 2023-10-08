# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Defines deprecated operators."""
import itertools
import numpy as np
from mindspore.common._decorator import deprecated
from mindspore import context
from mindspore import _checkparam as validator
from mindspore.ops import signature as sig
from mindspore.ops.primitive import Primitive, prim_attr_register
from mindspore.ops.operations.math_ops import _MathBinaryOp
from mindspore.ops.operations.nn_ops import _check_positive_int_or_tuple


class BNTrainingReduce(Primitive):
    """
    Please use BatchNorm instead.
    """
    @deprecated("1.5", "ops.BatchNorm", False)
    @prim_attr_register
    def __init__(self, data_format="NCHW"):
        """Initialize BNTrainingReduce."""
        super().__init__(name="BNTrainingReduce")
        self.init_prim_io_names(inputs=['x'], outputs=['sum', 'square_sum'])
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.name}', the 'NHWC' format is only supported in GPU target, "
                             f"but got the 'data_format' is {self.format} and "
                             f"the platform is {context.get_context('device_target')}.")
        self.add_prim_attr('data_format', self.format)


class BNTrainingUpdate(Primitive):
    """
    Please use BatchNorm instead.
    """
    @deprecated("1.5", "ops.BatchNorm", False)
    @prim_attr_register
    def __init__(self, isRef=True, epsilon=1e-5, factor=0.1, data_format="NCHW"):
        """Initialize BNTrainingUpdate."""
        super().__init__(name="BNTrainingUpdate")
        self.init_prim_io_names(inputs=['x', 'sum', 'square_sum', 'scale', 'b', 'mean', 'variance'],
                                outputs=['y', 'running_mean', 'running_variance', 'save_mean', 'save_inv_variance'])
        validator.check_value_type("isRef", isRef, [bool], self.name)
        validator.check_value_type("epsilon", epsilon, [float], self.name)
        validator.check_value_type("factor", factor, [float], self.name)
        self.epsilon = validator.check_float_range(epsilon, 0, 1, validator.INC_RIGHT, 'epsilon', 'BNTrainingUpdate')
        self.factor = validator.check_float_range(factor, 0, 1, validator.INC_BOTH, 'factor', 'BNTrainingUpdate')
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.name}', the 'NHWC' format is only supported in GPU target, "
                             f"but got the 'data_format' is {self.format} and "
                             f"the platform is {context.get_context('device_target')}.")
        self.add_prim_attr('data_format', self.format)


class MaxPoolWithArgmax(Primitive):
    """
    Please use MaxPoolWithArgmaxV2 instead.
    """
    @deprecated("2.0", "ops.MaxPoolWithArgmaxV2", False)
    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="valid", data_format="NCHW"):
        """Initialize MaxPoolWithArgmax."""
        super().__init__(name="MaxPoolWithArgmax")
        self.init_prim_io_names(inputs=['x'], outputs=['output', 'mask'])
        validator.check_value_type('kernel_size', kernel_size, [int, tuple], self.name)
        validator.check_value_type('strides', strides, [int, tuple], self.name)
        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.name)
        self.add_prim_attr("pad_mode", self.pad_mode)
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.name}', the 'NHWC' format is only supported in GPU target, "
                             f"but got the 'data_format' is {self.format} and "
                             f"the platform is {context.get_context('device_target')}.")
        self.kernel_size = _check_positive_int_or_tuple(
            "kernel_size", kernel_size, self.name, allow_four=False, ret_four=True)
        self.kernel_size = (1, self.kernel_size[-2], self.kernel_size[-1], 1)
        self.add_prim_attr("kernel_size", self.kernel_size)

        self.strides = _check_positive_int_or_tuple("strides", strides, self.name, allow_four=False, ret_four=True)
        self.strides = (1, self.strides[-2], self.strides[-1], 1)
        self.add_prim_attr("strides", self.strides)


class DropoutGenMask(Primitive):
    """
    Please use Dropout instead.
    """
    @deprecated("1.5", "ops.Dropout", False)
    @prim_attr_register
    def __init__(self, Seed0=0, Seed1=0):
        """Initialize DropoutGenMask."""
        super().__init__(name="DropoutGenMask")
        self.init_prim_io_names(inputs=['shape', 'keep_prob'], outputs=['output'])
        validator.check_value_type("Seed0", Seed0, [int], self.name)
        validator.check_value_type("Seed1", Seed1, [int], self.name)
        self.add_prim_attr("side_effect_hidden", True)


class DropoutDoMask(Primitive):
    """
    Please use Dropout instead.
    """
    @deprecated("1.5", "ops.Dropout", False)
    @prim_attr_register
    def __init__(self):
        super().__init__(name="DropoutDoMask")


class Gelu(Primitive):
    """
    Please use GeLU instead.
    """
    @deprecated("1.1", "GeLU", True)
    @prim_attr_register
    def __init__(self):
        """Initialize Gelu"""
        super().__init__(name="Gelu")
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class FastGelu(Primitive):
    """
    Please use FastGeLU instead.
    """
    @deprecated("1.1", "FastGeLU", True)
    @prim_attr_register
    def __init__(self):
        """Initialize FastGelu."""
        super().__init__(name="FastGelu")
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class TensorAdd(_MathBinaryOp):
    """
    Please use Add instead.
    """
    @deprecated("1.1", "Add", True)
    @prim_attr_register
    def __init__(self):
        """Initialize TensorAdd."""
        _MathBinaryOp.__init__(self)


class InplaceUpdate(Primitive):
    """
    Please use InplaceUpdateV2 instead.
    """
    @deprecated("2.0", "ops.InplaceUpdateV2", False)
    @prim_attr_register
    def __init__(self, indices):
        """Initialize InplaceUpdate"""
        self.init_prim_io_names(inputs=['x', 'v'], outputs=['y'])
        self.indices = indices
        validator.check_value_type("indices", indices, [int, tuple], self.name)
        if isinstance(indices, int):
            self.indices = (indices,)
        for item in self.indices:
            validator.check_value_type("item of indices", item, [int], self.name)


class DynamicShape(Primitive):
    """
    Please use TensorShape instead.
    """
    @deprecated("1.7", "TensorShape", True)
    @prim_attr_register
    def __init__(self, dtype=9):
        """init Shape"""
        super().__init__(name="DynamicShape")
        self.init_prim_io_names(inputs=['tensor'], outputs=['output'])
        self.add_prim_attr('is_dynamic_shape', True)


class GatherV2(Primitive):
    """
    Please use Gather instead.
    """
    @deprecated("1.1", "Gather", True)
    @prim_attr_register
    def __init__(self):
        """Initialize GatherV2"""
        super().__init__(name="GatherV2")
        self.add_prim_attr("batch_dims", 0)
        self.init_prim_io_names(inputs=['params', 'indices', 'axis'], outputs=['output'])


class ScalarToArray(Primitive):
    """
    Please use scalar_to_tensor instead.
    """
    @deprecated("2.0", "ops.scalar_to_tensor", False)
    @prim_attr_register
    def __init__(self):
        super().__init__(name="ScalarToArray")


class Pack(Primitive):
    """
    Please use Stack instead.
    """
    @deprecated("1.1", "Stack", True)
    @prim_attr_register
    def __init__(self, axis=0):
        """Initialize Pack"""
        super().__init__(name="Pack")
        validator.check_value_type("axis", axis, [int], self.name)
        self.axis = axis


class Unpack(Primitive):
    """
    Please use Unstack instead.
    """
    @deprecated("1.1", "Unstack", True)
    @prim_attr_register
    def __init__(self, axis=0):
        """Initialize Unpack"""
        super().__init__(name="Unpack")
        validator.check_value_type("axis", axis, [int], self.name)
        self.axis = axis


class ScatterNonAliasingAdd(Primitive):
    """
    Please use TensorScatterAdd instead.
    """
    __mindspore_signature__ = (
        sig.make_sig('input_x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )
    @deprecated("2.1", "ops.ScatterNonAliasingAdd", False)
    @prim_attr_register
    def __init__(self):
        """Initialize ScatterNonAliasingAdd"""
        super().__init__(name="ScatterNonAliasingAdd")
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)


class BatchToSpaceND(Primitive):
    """
    Please use batch_to_space_nd instead.
    """
    @deprecated("2.0", "ops.batch_to_space_nd", False)
    @prim_attr_register
    def __init__(self, block_shape, crops):
        """Initialize BatchToSpaceND"""
        super().__init__(name="BatchToSpaceND")
        if isinstance(block_shape, int):
            block_shape = (block_shape,) * np.array(crops).shape[0]
        self.add_prim_attr("block_shape", block_shape)
        validator.check_value_type('block_shape type', block_shape, [list, tuple], self.name)
        validator.check('block_shape shape', len(np.array(block_shape).shape), '', 1, validator.EQ, self.name)
        block_rank = len(block_shape)
        if context.get_context("device_target") == "Ascend":
            validator.check('block_shape length', block_rank, '', 2, validator.EQ, self.name)
        for elem in block_shape:
            validator.check('block_shape element', elem, '', 1, validator.GE, self.name)
            validator.check_value_type('block_shape element', elem, [int], self.name)
        self.block_shape = block_shape

        validator.check_value_type('crops type', crops, [list, tuple], self.name)
        validator.check('crops length', len(crops), '', 1, validator.GE, self.name)
        validator.check('crops shape', np.array(crops).shape, '', (block_rank, 2), validator.EQ, self.name)
        for elem in itertools.chain(*crops):
            validator.check_non_negative_int(elem, 'crops element', self.name)
            validator.check_value_type('crops element', elem, [int], self.name)
        self.crops = crops


class identity(Primitive):
    """
    Please use side_effect_propagate instead.
    """

    # Side effect will propagated from the first argument to return value.
    side_effect_propagate = 1

    @prim_attr_register
    def __init__(self):
        """Initialize identity."""
        super().__init__(name="identity")
        self.add_prim_attr('side_effect_propagate', 1)

    @deprecated('2.0', 'nn.Identity', False)
    def __call__(self, x):
        return x

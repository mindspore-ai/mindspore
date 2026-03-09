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
"""Utility helpers for operation testing.

This module provides:
- BinaryOpsFactory: utilities to compare MindSpore binary ops with Benchmark
  references.
- Static and dynamic-shape parity checks, with optional gradient comparisons.
"""
import functools
import pytest
import mindspore as ms
from mindspore.graph.api import _pynative_executor
from tests.st.ops.share._internal.meta import OpsFactory
from tests.st.ops.share._internal.utils import OpSampleInput, make_tensor
from tests.st.ops.share._op_info.op_info import BinaryOpInfo
from tests.st.ops.share._op_info.op_common import dtypes_as_torch, SMALL_DIM_SIZE

class BinaryOpsFactory(OpsFactory):
    """Factory for testing binary operations.

    It wires up sample inputs, reference functions, and gradient networks to
    run value and gradient parity checks between MindSpore and Benchmark.
    """
    def __init__(
            self,
            op_info: BinaryOpInfo,
            **kwargs,
    ):
        super().__init__(
            op_info,
            **kwargs,
        )
        # Ensure pylint knows _douts is defined in this class.
        self._douts = None

    def test_binary_op_tensor_type_promotion(self):
        """Test tensor-tensor type promotion across mixed dtypes.

        Builds pairs of tensors with different dtypes supported by both
        MindSpore and the benchmark backend, then compares forward results.
        """
        def sample_inputs_binary_tensor_type_promotion_func(op_info: BinaryOpInfo, dtypes, device=None):
            for input_dtype in dtypes:
                for other_dtype in dtypes:
                    if input_dtype == other_dtype:
                        continue

                    S = SMALL_DIM_SIZE
                    yield OpSampleInput(
                        op_input = make_tensor(shape=(S,S),
                                           dtype = input_dtype,
                                           low=op_info.input_low,
                                           high=op_info.input_high,
                                           exclude_value=op_info.input_exclude_value, ),
                        op_args = (make_tensor(shape=(S,S),
                                           dtype = other_dtype,
                                           low=op_info.other_low,
                                           high=op_info.other_high,
                                           exclude_value=op_info.other_exclude_value,),),
                        op_kwargs = {},
                        sample_name = f"{op_info.name}_{input_dtype}_2d_x_{other_dtype}_2d")

        try:
            if not self.op_info.support_tensor_type_promotion:
                print(f"\nop_name: {self.op_name} does not support tensor type promotion, "
                      f"skip test_binary_op_tensor_type_promotion.")
                return

            print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, "
                  f"test_binary_op_tensor_type_promotion...")
            self.supported_dtypes = tuple(set(self.supported_dtypes) & set(dtypes_as_torch))
            if self.op_info.promotion_dtypes_exclude is not None:
                self.supported_dtypes = tuple(set(self.supported_dtypes)
                                          - set(self.op_info.promotion_dtypes_exclude))
            self.op_basic_reference_inputs_func = sample_inputs_binary_tensor_type_promotion_func
            for sample_input in self.op_basic_reference_inputs_func(
                    self.op_info,
                    self.supported_dtypes,
                    device=self._device,
            ):
                self.compare_with_torch(sample_inputs=sample_input)
        except Exception as e:
            print(f"\ntest_binary_op_tensor_type_promotion failed:"
                  f"\nop_name: {self.op_name}"
                  f"\nmode: {self._context_mode}"
                  f"\ndtypes: {self.supported_dtypes}"
                  f"\n{sample_input.summary(True)}")
            raise e


    def test_binary_op_scalar_type_promotion(self):
        """Test scalar type promotion with tensor-scalar combinations.

        Covers left/right Python scalar, scalar tensors, and combinations across
        integer, floating, and complex dtypes based on support flags.
        """
        def test_tensor_x_scalar_promotion(input_dtype, other_dtype):
            if not {input_dtype, other_dtype}.issubset(set(self.supported_dtypes)):
                return
            try:
                _tensor = make_tensor_func(dtype=input_dtype,
                                           low=self.op_info.input_low,
                                           high=self.op_info.input_high,
                                           exclude_value=self.op_info.input_exclude_value)
                _scalar_tensor = make_scalar_tensor_func(dtype=other_dtype,
                                                         low=self.op_info.other_low,
                                                         high=self.op_info.other_high,
                                                         exclude_value=self.op_info.other_exclude_value)
                _sample_input = OpSampleInput(
                    op_input=_tensor,
                    op_args=(_scalar_tensor,),
                    op_kwargs={},
                    sample_name=(f"{self.op_info.name}_{input_dtype}_1d_x_{other_dtype}_0d"),
                )
                self.compare_with_torch_type_promotion(sample_inputs=_sample_input)
                if not other_dtype.is_complex and self.op_info.supports_right_python_scalar:
                    _sample_input = OpSampleInput(
                        op_input=_tensor,
                        op_args=(_scalar_tensor.item(),),
                        op_kwargs={},
                        sample_name=(f"{self.op_info.name}_{input_dtype}_1d_x_{other_dtype}_scalar"),
                    )
                    self.compare_with_torch_type_promotion(sample_inputs=_sample_input)
            except Exception as e:
                print(f"\test_tensor_x_scalar_promotionn failed:"
                      f"\nop_name: {self.op_name}"
                      f"\nmode: {self._context_mode}"
                      f"\n{_sample_input.summary(True)}")
                raise e

        def test_scalar_x_tensor_promotion(input_dtype, other_dtype):
            if not {input_dtype, other_dtype}.issubset(set(self.supported_dtypes)):
                return
            try:
                _scalar_tensor = make_scalar_tensor_func(dtype=input_dtype,
                                                         low=self.op_info.input_low,
                                                         high=self.op_info.input_high,
                                                         exclude_value=self.op_info.input_exclude_value)
                _tensor = make_tensor_func(dtype=other_dtype,
                                           low=self.op_info.other_low,
                                           high=self.op_info.other_high,
                                           exclude_value=self.op_info.other_exclude_value)

                _sample_input = OpSampleInput(
                    op_input=_scalar_tensor,
                    op_args=(_tensor,),
                    op_kwargs={},
                    sample_name=(f"{self.op_info.name}_{input_dtype}_0d_x_{other_dtype}_1d"),
                )
                self.compare_with_torch_type_promotion(sample_inputs=_sample_input)
                if not input_dtype.is_complex and self.op_info.supports_left_python_scalar:
                    _sample_input = OpSampleInput(
                        op_input=_scalar_tensor.item(),
                        op_args=(_tensor,),
                        op_kwargs={},
                        sample_name=(f"{self.op_info.name}_{input_dtype}_scalar_x_{other_dtype}_1d"),
                    )
                    self.compare_with_torch_type_promotion(sample_inputs=_sample_input)
            except Exception as e:
                print(f"\test_scalar_x_tensor_promotion failed:"
                      f"\nop_name: {self.op_name}"
                      f"\nmode: {self._context_mode}"
                      f"\n{_sample_input.summary(True)}")
                raise e

        def test_scalar_x_scalar_type_promotion(input_dtype, other_dtype):
            if not {input_dtype, other_dtype}.issubset(set(self.supported_dtypes)):
                return
            try:
                _scalar_tensor1 = make_scalar_tensor_func(dtype=input_dtype,
                                                          low=self.op_info.input_low,
                                                          high=self.op_info.input_high,
                                                          exclude_value=self.op_info.input_exclude_value)
                _scalar_tensor2 = make_scalar_tensor_func(dtype=other_dtype,
                                                          low=self.op_info.other_low,
                                                          high=self.op_info.other_high,
                                                          exclude_value=self.op_info.other_exclude_value)
                if input_dtype not in [ms.complex64, ms.complex128]:
                    _sample_input = OpSampleInput(
                        op_input=_scalar_tensor1.item(),
                        op_args=(_scalar_tensor2,),
                        op_kwargs={},
                        sample_name=(f"{self.op_info.name}_{input_dtype}_scalar_x_{other_dtype}_0d"),
                    )
                    self.compare_with_torch(sample_inputs=_sample_input)
                if other_dtype not in [ms.complex64, ms.complex128]:
                    _sample_input = OpSampleInput(
                        op_input=_scalar_tensor1,
                        op_args=(_scalar_tensor2.item(),),
                        op_kwargs={},
                        sample_name=(f"{self.op_info.name}_{input_dtype}_0d_x_{other_dtype}_scalar"),
                    )
                    self.compare_with_torch(sample_inputs=_sample_input)
                if (not input_dtype.is_complex and not other_dtype.is_complex
                        and self.op_info.supports_both_python_scalar):
                    _sample_input = OpSampleInput(
                        op_input=_scalar_tensor1.item(),
                        op_args=(_scalar_tensor2.item(),),
                        op_kwargs={},
                        sample_name=(f"{self.op_info.name}_{input_dtype}_scalar_x_{other_dtype}_scalar"),
                    )
                    self.compare_with_torch(sample_inputs=_sample_input)
            except Exception as e:
                print(f"\test_scalar_x_scalar_type_promotion failed:"
                      f"\nop_name: {self.op_name}"
                      f"\nmode: {self._context_mode}"
                      f"\n{_sample_input.summary(True)}")
                raise e

        S = SMALL_DIM_SIZE
        if not self.op_info.support_tensor_type_promotion:
            print(f"\nop_name: {self.op_name} does not support tensor type promotion, "
                  f"skip test_binary_op_scalar_type_promotion.")
            return
        make_tensor_func = functools.partial(make_tensor, shape=(S,), device=self._device)
        make_scalar_tensor_func = functools.partial(make_tensor, shape=(), device=self._device)
        for fn, flag in {test_tensor_x_scalar_promotion:self.op_info.supports_right_python_scalar,
                         test_scalar_x_tensor_promotion:self.op_info.supports_left_python_scalar,
                         test_scalar_x_scalar_type_promotion:self.op_info.supports_both_python_scalar}.items():
            if flag:
                for input_dtype in self.supported_dtypes:
                    for other_dtype in self.supported_dtypes:
                        if input_dtype == other_dtype:
                            continue
                        fn(input_dtype, other_dtype)

    def test_binary_op_error(self):
        '''
        Test binary op error cases.
        '''
        if self.op_error_inputs_func is None:
            print(f"\nop_name: {self.op_name} has no op_error_inputs_func, "
                  f"skip test_binary_op_error.")
            return

        try:
            print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, test_binary_op_error...")
            for error_input in self.op_error_inputs_func(self.op_info, device=self._device):
                _sample_input = error_input.op_sample_input
                _error_type = error_input.op_error_type

                with pytest.raises(_error_type):
                    self.op(_sample_input.op_input, *_sample_input.op_args, **_sample_input.op_kwargs)
                    _pynative_executor.sync()
        except Exception as e:
            print(f"\ntest_binary_op_error catch expect {_error_type.__name__} failed, but got {type(e).__name__}:"
                  f"\nop_name: {self.op_name}"
                  f"\nmode: {self._context_mode}"
                  f"\nsample_input: {_sample_input.summary()}"
                  f"\nerror_info: {error_input.op_error_info}")
            raise e

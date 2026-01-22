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
from tests.st.ops.share._op_info.op_info import OpInfo
from tests.st.ops.share._op_info.op_common import dtypes_as_torch, SMALL_DIM_SIZE


class BinaryOpsFactory(OpsFactory):
    """Factory for testing binary operations.

    It wires up sample inputs, reference functions, and gradient networks to
    run value and gradient parity checks between MindSpore and Benchmark.
    """
    def __init__(
            self,
            op_info: OpInfo,
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
        def sample_inputs_binary_tensor_type_promotion_func(op_info: OpInfo, dtypes, device=None):
            for input_dtype in dtypes:
                for other_dtype in dtypes:
                    if input_dtype == other_dtype:
                        continue

                    S = SMALL_DIM_SIZE
                    yield OpSampleInput(
                            op_input=make_tensor(shape=(S, S), dtype=input_dtype, device=device),
                            op_args=(make_tensor(shape=(S, S), dtype=other_dtype, device=device),),
                            op_kwargs={},
                            sample_name=op_info.name,
                        )
        try:
            if not self.op_info.support_tensor_type_promotion:
                print(f"\nop_name: {self.op_name} does not support tensor type promotion, "
                      f"skip test_binary_op_tensor_type_promotion.")
                return

            print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, "
                  f"test_binary_op_tensor_type_promotion...")
            self.supported_dtypes = tuple(set(self.supported_dtypes) & set(dtypes_as_torch))
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
        def test_intxfloat_scalar_type_promotion(right_python_scalar=True):
            """int64 tensor x float scalar(type/tensor) promotion cases.

            Args:
                right_python_scalar (bool): If True, place scalar on the right;
                    otherwise place scalar on the left.
            """
            if not {ms.int64, ms.float32}.issubset(set(self.supported_dtypes)):
                return

            try:
                print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, "
                      f"{'right_python_scalar' if right_python_scalar else 'left_python_scalar'}, "
                      f"test_intxfloat_scalar_type_promotion...")
                # int tensor x float scalar
                _tensor = make_tensor_func(dtype=ms.int64)
                _scalar = 1.0
                _sample_input = OpSampleInput(
                    op_input=_tensor if right_python_scalar else _scalar,
                    op_args=(_scalar,) if right_python_scalar else (_tensor,),
                    op_kwargs={},
                    sample_name=self.op_info.name,
                )
                self.compare_with_torch(sample_inputs=_sample_input)
                # int tensor x float scalar tensor
                _scalar_tensor = make_scalar_tensor_func(dtype=ms.float32)
                _sample_input = OpSampleInput(
                    op_input=_tensor if right_python_scalar else _scalar_tensor,
                    op_args=(_scalar_tensor,) if right_python_scalar else (_tensor,),
                    op_kwargs={},
                    sample_name=self.op_info.name,
                )
                self.compare_with_torch(sample_inputs=_sample_input)
                # int tensor x float64 scalar tensor
                if ms.float64 in self.supported_dtypes:
                    _scalar_tensor = make_scalar_tensor_func(dtype=ms.float64)
                    _sample_input = OpSampleInput(
                        op_input=_tensor if right_python_scalar else _scalar_tensor,
                        op_args=(_scalar_tensor,) if right_python_scalar else (_tensor,),
                        op_kwargs={},
                        sample_name=self.op_info.name,
                    )
                    self.compare_with_torch(sample_inputs=_sample_input)
            except Exception as e:
                print(f"\test_intxfloat_scalar_type_promotion failed:"
                      f"\nop_name: {self.op_name}"
                      f"\nmode: {self._context_mode}"
                      f"\n{'right_python_scalar' if right_python_scalar else 'left_python_scalar'}, "
                      f"\n{_sample_input.summary(True)}")
                raise e

        def test_floatxcomplex_scalar_type_promotion(right_python_scalar=True):
            """float32 tensor x complex scalar(tensor) promotion cases.

            Args:
                right_python_scalar (bool): If True, place scalar on the right;
                    otherwise place scalar on the left.
            """
            if not {ms.float32, ms.complex64}.issubset(set(self.supported_dtypes)):
                return

            try:
                print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, "
                      f"{'right_python_scalar' if right_python_scalar else 'left_python_scalar'}, "
                      f"test_floatxcomplex_scalar_type_promotion...")
                # float tensor x complex64 scalar tensor
                _tensor = make_tensor_func(dtype=ms.float32)
                _scalar_tensor = make_scalar_tensor_func(dtype=ms.complex64)
                _sample_input = OpSampleInput(
                    op_input=_tensor if right_python_scalar else _scalar_tensor,
                    op_args=(_scalar_tensor,) if right_python_scalar else (_tensor,),
                    op_kwargs={},
                    sample_name=self.op_info.name,
                )
                self.compare_with_torch(sample_inputs=_sample_input)
                # float tensor x complex128 scalar tensor
                if ms.complex128 in self.supported_dtypes:
                    _scalar_tensor = make_scalar_tensor_func(dtype=ms.complex128)
                    _sample_input = OpSampleInput(
                        op_input=_tensor if right_python_scalar else _scalar_tensor,
                        op_args=(_scalar_tensor,) if right_python_scalar else (_tensor,),
                        op_kwargs={},
                        sample_name=self.op_info.name,
                    )
                    self.compare_with_torch(sample_inputs=_sample_input)
            except Exception as e:
                print(f"\test_floatxcomplex_scalar_type_promotion failed:"
                      f"\nop_name: {self.op_name}"
                      f"\nmode: {self._context_mode}"
                      f"\n{'right_python_scalar' if right_python_scalar else 'left_python_scalar'}, "
                      f"\n{_sample_input.summary(True)}")
                raise e

        def test_floatxfloat_scalar_type_promotion(right_python_scalar=True):
            """float32 tensor x float64 scalar(tensor) promotion cases.

            Args:
                right_python_scalar (bool): If True, place scalar on the right;
                    otherwise place scalar on the left.
            """
            if not {ms.float32, ms.float64}.issubset(set(self.supported_dtypes)):
                return

            try:
                print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, "
                      f"{'right_python_scalar' if right_python_scalar else 'left_python_scalar'}, "
                      f"test_floatxfloat_scalar_type_promotion...")
                _tensor = make_tensor_func(dtype=ms.float32)
                _scalar_tensor = make_scalar_tensor_func(dtype=ms.float64)
                _sample_input = OpSampleInput(
                    op_input=_tensor if right_python_scalar else _scalar_tensor,
                    op_args=(_scalar_tensor,) if right_python_scalar else (_tensor,),
                    op_kwargs={},
                    sample_name=self.op_info.name,
                )
                self.compare_with_torch(sample_inputs=_sample_input)
            except Exception as e:
                print(f"\test_floatxfloat_scalar_type_promotion failed:"
                      f"\nop_name: {self.op_name}"
                      f"\nmode: {self._context_mode}"
                      f"\n{'right_python_scalar' if right_python_scalar else 'left_python_scalar'}, "
                      f"\n{_sample_input.summary(True)}")
                raise e

        def test_complexxcomplex_scalar_type_promotion(right_python_scalar=True):
            """complex64 tensor x complex128 scalar(tensor) promotion cases.

            Args:
                right_python_scalar (bool): If True, place scalar on the right;
                    otherwise place scalar on the left.
            """
            if not {ms.complex64, ms.complex128}.issubset(set(self.supported_dtypes)):
                return

            try:
                print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, "
                      f"{'right_python_scalar' if right_python_scalar else 'left_python_scalar'}, "
                      f"test_complexxcomplex_scalar_type_promotion...")
                # complex64 tensor x complex128 scalar tensor
                _tensor = make_tensor_func(dtype=ms.complex64)
                _scalar_tensor = make_scalar_tensor_func(dtype=ms.complex128)
                _sample_input = OpSampleInput(
                    op_input=_tensor if right_python_scalar else _scalar_tensor,
                    op_args=(_scalar_tensor,) if right_python_scalar else (_tensor,),
                    op_kwargs={},
                    sample_name=self.op_info.name,
                )
                self.compare_with_torch(sample_inputs=_sample_input)
            except Exception as e:
                print(f"\test_complexxcomplex_scalar_type_promotion failed:"
                      f"\nop_name: {self.op_name}"
                      f"\nmode: {self._context_mode}"
                      f"\n{'right_python_scalar' if right_python_scalar else 'left_python_scalar'}, "
                      f"\n{_sample_input.summary(True)}")
                raise e

        def test_all_scalar_type_promotion():
            """Pure Python scalar x scalar promotion cases (int/float)."""
            if not {ms.int64, ms.float32}.issubset(set(self.supported_dtypes)):
                return

            try:
                print(f"\nop_name: {self.op_name}, mode:{self._context_mode},"
                      f"test_all_scalar_type_promotion...")
                # int scalar x float scalar
                _r_scalar = 2.0
                for _l_scalar in (1, 1.0):
                    _sample_input = OpSampleInput(
                        op_input=_l_scalar,
                        op_args=(_r_scalar,),
                        op_kwargs={},
                        sample_name=self.op_info.name,
                    )
                    self.compare_with_torch(sample_inputs=_sample_input)
            except Exception as e:
                print(f"\test_all_scalar_type_promotion failed:"
                      f"\nop_name: {self.op_name}"
                      f"\nmode: {self._context_mode}"
                      f"\n{_sample_input.summary(True)}")
                raise e

        S = SMALL_DIM_SIZE
        make_tensor_func = functools.partial(make_tensor, shape=(S,), device=self._device)
        make_scalar_tensor_func = functools.partial(make_tensor, shape=(), device=self._device)
        if self.op_info.supports_right_python_scalar:
            test_intxfloat_scalar_type_promotion()
            test_floatxcomplex_scalar_type_promotion()
            test_floatxfloat_scalar_type_promotion()
            test_complexxcomplex_scalar_type_promotion()
        if self.op_info.supports_left_python_scalar:
            test_intxfloat_scalar_type_promotion(right_python_scalar=False)
            test_floatxcomplex_scalar_type_promotion(right_python_scalar=False)
            test_floatxfloat_scalar_type_promotion(right_python_scalar=False)
            test_complexxcomplex_scalar_type_promotion(right_python_scalar=False)
        if self.op_info.supports_both_python_scalar:
            test_all_scalar_type_promotion()

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

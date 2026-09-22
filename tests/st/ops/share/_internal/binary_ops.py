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
import pytest
import mindspore as ms
from mindspore.graph.api import _pynative_executor
from tests.st.ops.share._internal.meta import OpsFactory
from tests.st.ops.share._op_info.op_info import BinaryOpInfo
from tests.st.ops.share._op_info.op_common import dtypes_as_torch

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
            # The generator may raise before yielding, so keep the name bound for the report below.
            sample_input = None
            for sample_input in self.op_info.op_inputs_tensor_type_promotion_func(
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
                  f"\n{sample_input.summary(True) if sample_input is not None else '<sample generation failed>'}")
            raise e


    def test_binary_op_scalar_type_promotion(self):
        """Test scalar type promotion with tensor-scalar combinations.

        Covers left/right Python scalar, scalar tensors, and combinations across
        integer, floating, and complex dtypes based on support flags.
        """

        if not self.op_info.support_tensor_type_promotion:
            print(f"\nop_name: {self.op_name} does not support tensor type promotion, "
                  f"skip test_binary_op_scalar_type_promotion.")
            return

        def get_test_name(sample_input):
            input_tensor=sample_input.op_input
            other_tensor=sample_input.op_args[0]
            if isinstance(input_tensor, ms.Tensor) and isinstance(other_tensor, ms.Tensor):
                return 'test_scalar_x_tensor_promotion' if len(input_tensor.shape)==0 else \
                    'test_tensor_x_scalar_promotion'
            return 'test_scalar_x_scalar_type_promotion'

        for sample_input in self.op_info.op_inputs_scalar_type_promotion_func(self.op_info,
                    self.supported_dtypes,
                    device=self._device,):
            try:
                self.compare_with_torch_type_promotion(sample_inputs=sample_input)
            except Exception as e:
                print(f"\n{get_test_name(sample_input)} failed:"
                      f"\nop_name: {self.op_name}"
                      f"\nmode: {self._context_mode}"
                      f"\n{sample_input.summary(True)}")
                raise e



    def test_binary_op_error(self):
        '''
        Test binary op error cases.
        '''
        if self.op_error_inputs_func is None:
            print(f"\nop_name: {self.op_name} has no op_error_inputs_func, "
                  f"skip test_binary_op_error.")
            return
        if self.should_skip_backend_dtype_coverage("test_binary_op_error"):
            return

        print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, test_binary_op_error...")
        for error_input in self.op_error_inputs_func(self.op_info, device=self._device):
            _sample_input = error_input.op_sample_input
            _error_type = error_input.op_error_type

            try:
                self.op(_sample_input.op_input, *_sample_input.op_args, **_sample_input.op_kwargs)
                _pynative_executor.sync()
            except _error_type:
                continue
            else:
                e_type = list(map(lambda x: x.__name__, _error_type if isinstance(_error_type,
                                                                                  (list, tuple)) else [_error_type]))
                print(f"\ntest_binary_op_error catch expect {e_type} failed:"
                      f"\nop_name: {self.op_name}"
                      f"\nmode: {self._context_mode}"
                      f"\nsample_input: {_sample_input.summary()}"
                      f"\nerror_info: {error_input.op_error_info}")
                pytest.fail("没有抛出预期异常")

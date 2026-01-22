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
- OpsFactory: a base test factory handling context/device, sample inputs,
  and comparisons.
- Helper networks: forward/grad nets such as OpsCommonNet, OpsCommonNetNoKwargs,
  OpCommonGradNetFirstInput, and OpCommonGradNetAllInput.
- Comparison routines: static and dynamic-shape forward/grad parity checks
  against reference backends.
"""
# pylint: disable=R1705
import warnings
import torch
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore._c_expression import MSContext
from mindspore.common.dtype import _dtype_to_nptype
from mindspore.graph.api import _pynative_executor
from typing import Optional, Union, List, final
from tests.st.utils.test_utils import single_golden_compare, double_golden_compare, OpTypes
from tests.st.ops.share._internal.utils import (
    OpSampleInput, OpDynamicInput, is_op_input_dynamic, make_tensor, ms_asnumpy, torch_asnumpy
)
from tests.st.ops.share._op_info.op_info import OpInfo
from tests.st.ops.share._op_info.op_common import get_default_loss, dtypes_extra_uint


@ms.jit(backend="ms_backend")
def ops_common_net(op, op_input, *op_args, **op_kwargs):
    """Forward op net wrapper with jit.
    """
    return op(op_input, *op_args, **op_kwargs)


class OpCommonGradNetAllInput(nn.Cell):
    """Gradient network for all inputs.

    Before use, ensure op_kwargs are converted to op_args using
    OpSampleInput.convert_to_args() and append dout to op_args.
    """
    def __init__(self, op, *, grad_position):
        super().__init__()
        self.grad = ms.grad(op, grad_position=grad_position)

    def construct(self, *op_args):
        return self.grad(*op_args)


class OpsFactory():
    """Base test factory for operators.

    Manages device/context, builds sample inputs, forwards through MindSpore
    and references, and performs value/gradient comparisons.
    """
    def __init__(
            self,
            op_info: OpInfo,
            **kwargs,
    ):
        self.op_info = op_info
        # inner params
        self._douts = None
        self._device = None
        self._ascend_name = None
        self._context_mode = 'pynative'
        self._op_net_func = ops_common_net
        self._op_grad_func = None
        self._op_grad_cell = OpCommonGradNetAllInput

        self._parse_op_info(self.op_info)

    @final
    def _parse_op_info(self, op_info: OpInfo):
        """Populate factory fields from `OpInfo` and current device context.

        Args:
            op_info: Operator metadata including op callable, reference, dtypes,
                sample input builder, compare method, etc.
        """
        self.op = op_info.op
        self.customize_op_func = op_info.customize_op_func
        self.ref = op_info.ref
        self.op_name = op_info.name
        self.op_basic_reference_inputs_func = op_info.op_basic_reference_inputs_func
        self.op_extra_reference_inputs_func = op_info.op_extra_reference_inputs_func
        self.op_dynamic_inputs_func = op_info.op_dynamic_inputs_func
        self.op_error_inputs_func = op_info.op_error_inputs_func
        self.grad_position = op_info.grad_position
        self.flatten_inputs_for_backward = op_info.flatten_inputs_for_backward
        self._sample_inputs = None
        self._dynamic_inputs = None

        # get supported dtypes for the op with entire environment.
        device = ms.context.get_context('device_target').lower()
        if device == 'ascend':
            self._ascend_name = MSContext.get_instance().get_ascend_soc_version()
            if self._ascend_name == 'ascend910b':
                self.supported_dtypes = op_info.dtypes_ascend910b
            else:
                self.supported_dtypes = op_info.dtypes_ascend
        elif device == 'cpu':
            self.supported_dtypes = op_info.dtypes_cpu
        elif device == 'gpu':
            self.supported_dtypes = op_info.dtypes_gpu
        else:
            raise ValueError(f"Invalid device: {device}, expected: 'ascend', 'cpu', 'gpu'.")

        self._device = device
        self._inplace_op = getattr(op_info, 'is_inplace_op', False)
        # op of torch don't support extra uint dtypes, so set convert_extra_uint to True if mindspore supports them.
        self._convert_extra_uint = bool(set(self.supported_dtypes) & set(dtypes_extra_uint))

        self._convert_half_to_float = getattr(op_info, 'convert_half_to_float', False)
        if not self._convert_half_to_float:
            # if op does not support float16 on certain backend of benchmark,
            # such as sum of torch gpu can't support float16.
            # the float16 will be converted to float32 for benchmark calculation,
            # and convert back to float16 for comparison. op of torch gpu don't support float16 usually.
            self._convert_half_to_float = device == 'gpu'

        self._compare_method = op_info.compare_method
        self._default_golden_loss_func = op_info.default_golden_loss_func
        self._ascend_forward_loss_override = op_info.ascend_forward_loss_override
        self._ascend910b_forward_loss_override = op_info.ascend910b_forward_loss_override
        self._cpu_forward_loss_override = op_info.cpu_forward_loss_override
        self._gpu_forward_loss_override = op_info.gpu_forward_loss_override
        self._ascend_backward_loss_override = op_info.ascend_backward_loss_override
        self._ascend910b_backward_loss_override = op_info.ascend910b_backward_loss_override
        self._cpu_backward_loss_override = op_info.cpu_backward_loss_override
        self._gpu_backward_loss_override = op_info.gpu_backward_loss_override

    @final
    def _generate_random_dout(self, return_torch_douts=False):
        """Generate random dout tensors for the op.

        Args:
            return_torch_douts (bool): Whether to return Pytorch douts.

        Returns:
            list | None: Random douts or None when not requested.
        """

        if self._douts is None:
            def _make_ms_dout_for_output(out_obj):
                # Create MindSpore sens matching output structure for multi-output ops.
                if isinstance(out_obj, (tuple, list)):
                    return tuple(make_tensor(o.shape, o.dtype, random_method='randn') for o in out_obj)
                return make_tensor(out_obj.shape, out_obj.dtype, random_method='randn')

            ms_out = self.forward_mindspore_impl()
            self._douts = [_make_ms_dout_for_output(outi) for outi in ms_out]

        if return_torch_douts:
            def _to_torch(d):
                if isinstance(d, (tuple, list)):
                    converted = tuple(torch.tensor(ms_asnumpy(x)) for x in d)
                    if self._convert_half_to_float:
                        converted = tuple(x.float() if x.dtype == torch.float16 else x for x in converted)
                    return converted
                t = torch.tensor(ms_asnumpy(d))
                if self._convert_half_to_float and t.dtype == torch.float16:
                    t = t.float()
                return t

            return [_to_torch(d) for d in self._douts]
        return None

    @final
    def update_op_net_func(
            self,
            *,
            op_net_func=None,
            op_grad_func=None,
            op_grad_cell=None
    ):
        """Update forward/grad network wrappers used by the factory.

        Args:
            op_net_func: Net class for standard forward execution.
            op_grad_func: Function for gradient computation.
        """
        self._op_net_func = op_net_func if op_net_func is not None else self._op_net_func
        self._op_grad_func = op_grad_func if op_grad_func is not None else self._op_grad_func
        self._op_grad_cell = op_grad_cell if op_grad_cell is not None else self._op_grad_cell

    @final
    def _get_current_loss_override(self, *, grad_cmp: bool = False):
        """Select override loss dict based on device/backend and forward/backward.
        
        Args:
            grad_cmp: True if comparing gradients (backward), else forward.
        Returns:
            dict | None: The selected override mapping for dtype->loss, or None.
        """
        device = self._device
        if device == 'ascend':
            if self._ascend_name == 'ascend910b':
                return self._ascend910b_backward_loss_override if grad_cmp else self._ascend910b_forward_loss_override
            return self._ascend_backward_loss_override if grad_cmp else self._ascend_forward_loss_override
        if device == 'cpu':
            return self._cpu_backward_loss_override if grad_cmp else self._cpu_forward_loss_override
        if device == 'gpu':
            return self._gpu_backward_loss_override if grad_cmp else self._gpu_forward_loss_override
        return None

    @final
    def update_inputs(
            self,
            op_sample_inputs: Union[List[OpSampleInput], OpSampleInput] = None,
            op_dynamic_inputs: OpDynamicInput = None,
    ):
        """Update the sample and dynamic inputs.

        Args:
            op_sample_inputs: OpSampleInput object.
            op_dynamic_inputs: OpDynamicInput object.
        """
        if op_sample_inputs is not None:
            self._sample_inputs = op_sample_inputs
        if op_dynamic_inputs is not None:
            self._dynamic_inputs = op_dynamic_inputs

    @final
    def set_context_mode(
            self,
            *,
            mode=None
    ):
        """Set the execution context mode for the op.

        Args:
            mode: One of 'kbk', 'ge', 'pynative', or a MindSpore mode enum.
        """
        if mode is not None:
            if isinstance(mode, str):
                if mode.lower() == 'kbk':
                    ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
                elif mode.lower() == 'ge':
                    ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')
                elif mode.lower() == 'pynative':
                    ms.context.set_context(mode=ms.PYNATIVE_MODE)
                else:
                    raise ValueError(f"Invalid mode: {mode}, expected: 'kbk', 'ge', 'pynative'.")
            else:
                ms.context.set_context(mode=mode)
            self._context_mode = mode

    @final
    def assert_equal(
            self,
            actual,
            expect,
            rtol=None,
            atol=None,
            *,
            compare_method='default_golden',
            ksize=None,
            op_type=None,
            secend_expect=None,
            grad_cmp=False,
    ):
        """Assert equality within tolerances using configured comparison.

        Args:
            actual: Actual output.
            expect: Expected output.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            compare_method: 'default_golden' | 'single_golden' | 'double_golden'.
            ksize: Kernel size for certain comparisons.
            op_type: Operation type enum for golden comparisons.
            secend_expect: Second expected output (for double golden).

        Note:
            Override to plug in other comparison strategies if needed.
        """
        def _count_unequal_element(expect, actual, rtol, atol):
            assert expect.shape == actual.shape
            total_count = len(expect.flatten())
            error = np.abs(expect - actual)
            greater = np.greater(error, atol + np.abs(actual) * rtol)
            nan_diff = np.not_equal(np.isnan(expect), np.isnan(actual))
            inf_diff = np.not_equal(np.isinf(expect), np.isinf(actual))
            if expect.dtype in ('complex64', 'complex128'):
                greater = greater + nan_diff + inf_diff
            else:
                neginf_diff = np.not_equal(np.isneginf(expect), np.isneginf(actual))
                greater = greater + nan_diff + inf_diff + neginf_diff
            loss_count = np.count_nonzero(greater)
            assert (loss_count / total_count) < rtol, \
                "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
                    format(expect[greater], actual[greater], error[greater])

        def allclose_nparray(expect, actual, rtol, atol, equal_nan=True):
            if not np.allclose(expect, actual, rtol, atol, equal_nan=equal_nan):
                _count_unequal_element(expect, actual, rtol, atol)
            else:
                assert np.array(expect).shape == np.array(actual).shape

        def default_golden_compare(expect, actual, rtol, atol):
            def convert_tensor_to_nparray(tensor):
                if isinstance(tensor, torch.Tensor):
                    return tensor.float().cpu().detach().numpy() \
                        if tensor.dtype == torch.bfloat16 else tensor.cpu().detach().numpy()
                if isinstance(tensor, ms.Tensor):
                    return ms_asnumpy(tensor)
                return tensor

            #910A will probably exceed limit 65504 at float16 or 3.4028235e+38 at float32 while calculate some
            #operators， mindspore will generate 65504/3.4028235e+38 while torch offer inf
            def _convert_910a_float_limit_to_inf(actual, expect, actual_dtype):
                if self._device=='ascend' and MSContext.get_instance().get_ascend_soc_version() == "ascend910":
                    if actual_dtype==ms.float16 and isinstance(expect, torch.Tensor):
                        actual = ops.where(ops.logical_and(actual >= 65504,
                                 ms.Tensor(torch_asnumpy(torch.isinf(expect)))),
                                 ms.Tensor(np.inf, dtype=actual_dtype),actual)
                        actual = ops.where(ops.logical_and(actual <= -65504,
                                        ms.Tensor(torch_asnumpy(torch.isneginf(expect)))),
                                           ms.Tensor(-np.inf, dtype=actual_dtype), actual)
                    elif actual_dtype==ms.float32 and isinstance(expect, torch.Tensor):
                        actual = ops.where(ops.logical_and(actual >= 3.4028235e+38,
                                        ms.Tensor(torch_asnumpy(torch.isinf(expect)))),
                                           ms.Tensor(np.inf, dtype=actual_dtype), actual)
                        actual = ops.where(ops.logical_and(actual <= -3.4028235e+38,
                                        ms.Tensor(torch_asnumpy(torch.isneginf(expect)))),
                                           ms.Tensor(-np.inf, dtype=actual_dtype), actual)
                return actual
            if isinstance(actual, (ms.Tensor, torch.Tensor, np.ndarray)):
                actual_dtype = actual.dtype
            else:
                actual_dtype = type(actual)

            loss_override_map = self._get_current_loss_override(grad_cmp=grad_cmp)
            if loss_override_map and actual_dtype in loss_override_map.keys():
                loss = loss_override_map[actual_dtype]
            elif self._default_golden_loss_func:
                loss = self._default_golden_loss_func(actual_dtype)
            else:
                loss = get_default_loss(actual_dtype)

            rtol = loss if rtol is None else rtol
            atol = loss if atol is None else atol

            actual = _convert_910a_float_limit_to_inf(actual, expect, actual_dtype)
            actual = convert_tensor_to_nparray(actual)
            expect = convert_tensor_to_nparray(expect)

            if self._convert_extra_uint:
                # Normalize dtype pairs when one side has been cast to int64 for
                # extra uint compatibility. Make membership explicit and stable.
                extra_uint_np_dtypes = tuple(map(_dtype_to_nptype, dtypes_extra_uint))
                if actual.dtype in extra_uint_np_dtypes and expect.dtype == np.int64:
                    expect = expect.astype(actual.dtype)
                if expect.dtype in extra_uint_np_dtypes and actual.dtype == np.int64:
                    actual = actual.astype(expect.dtype)

            allclose_nparray(expect, actual, rtol, atol)

        def convert_mindspore_extra_uint_dtype_to_int64(tensor):
            extra_uint_dtypes = [ms.uint16, ms.uint32, ms.uint64]
            if isinstance(tensor, ms.Tensor) and tensor.dtype in extra_uint_dtypes:
                return tensor.to(ms.int64)
            return tensor

        def convert_torch_float_to_half(x, y):
            if isinstance(x, ms.Tensor) and x.dtype == ms.float16:
                if isinstance(y, torch.Tensor) and y.dtype == torch.float32:
                    y = y.to(torch.float16)
            if isinstance(y, ms.Tensor) and y.dtype == ms.float16:
                if isinstance(x, torch.Tensor) and x.dtype == torch.float32:
                    x = x.to(torch.float16)
            return x, y

        if self._convert_extra_uint and compare_method != 'default_golden':
            expect = convert_mindspore_extra_uint_dtype_to_int64(expect)
            actual = convert_mindspore_extra_uint_dtype_to_int64(actual)

        if self._convert_half_to_float:
            expect, actual = convert_torch_float_to_half(expect, actual)

        if compare_method == 'default_golden':
            default_golden_compare(expect, actual, rtol, atol)
        elif compare_method == 'single_golden':
            assert single_golden_compare(expect, actual, ksize, op_type)
        elif compare_method == 'double_golden':
            assert double_golden_compare(expect, secend_expect, actual, op_type)
        else:
            raise ValueError(f"Invalid compare_method: {compare_method}, expected: 'default_golden', 'single_golden', \
                              'double_golden'.")

    def forward_mindspore_impl(
            self,
            *args,
            **kwargs
    ):
        """Run forward with the MindSpore implementation.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Outputs per sample input.
        """
        out = []
        for sample_input in self._sample_inputs:
            if self._inplace_op:
                sample_input = sample_input.copy()
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs
            if isinstance(op_input, ms.Tensor) and not op_input.dtype.is_complex:
                op_input = op_input.copy()
            op_args = [op_arg.copy() if isinstance(op_arg, ms.Tensor) and not op_arg.dtype.is_complex else op_arg \
                       for op_arg in op_args]

            if self._context_mode == 'pynative':
                outi = self.op(op_input, *op_args, **op_kwargs)
            else:
                outi = self._op_net_func(self.op, op_input, *op_args, **op_kwargs)
            out.append(outi)

        _pynative_executor.sync()
        return out

    def forward_pytorch_impl(
            self,
            *args,
            **kwargs
    ):
        """Run forward with the PyTorch reference implementation.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Outputs per sample input.
        """
        torch_fn = self.ref
        out = []

        for sample_input in self._sample_inputs:
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.astorch(convert_half_to_float=self._convert_half_to_float,
                                                convert_extra_uint=self._convert_extra_uint,
                            convert_scalar_to_tensor=getattr(self.op_info, 'convert_ref_scalar_to_tensor',False))
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs
            outi = torch_fn(op_input, *op_args, **op_kwargs)
            out.append(outi)

        return out

    def forward_tensorflow_impl(
            self,
            *args,
            **kwargs
    ):
        """Run forward with the TensorFlow reference implementation.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).
        """
        raise NotImplementedError

    def forward_numpy_impl(
            self,
            *args,
            **kwargs
    ):
        """Run forward with the NumPy reference implementation.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Outputs per sample input.
        """
        np_fn = self.ref
        out = []

        for sample_input in self._sample_inputs:
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.asnumpy()
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs

            outi = np_fn(op_input, *op_args, **op_kwargs)
            out.append(outi)

        return out

    def grad_mindspore_impl(
            self,
            *args,
            **kwargs
    ):
        """Compute gradients with the MindSpore implementation.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Gradients per sample input.
        """
        # TODO: use customized dout when ms.grad supports dout input
        #self._douts = None
        #self._generate_random_dout()

        grad_func = self._op_grad_func
        grads = []

        def _ms_tensor_supports_grad(t):
            return isinstance(t, ms.Tensor) and (t.dtype.is_floating_point or t.dtype.is_complex)

        for sample_input in self._sample_inputs:
            if self._inplace_op:
                sample_input = sample_input.copy()
            # No-dout args for indexing; with-dout args for actual grad call
            args_no_dout = sample_input.convert_to_args(flatten_op_input=self.flatten_inputs_for_backward).op_args
            # Use convert_to_args to append dout as a single positional argument (supports multi-output sens)
            #args_with_dout = sample_input.convert_to_args(append_dout=self._douts[idx]).op_args
            # get grad_position (must be int or tuple) and instantiate grad_func
            tensor_indices = tuple(i for i, v in enumerate(args_no_dout) if _ms_tensor_supports_grad(v))
            if not tensor_indices:
                grads.append(tuple())
                warnings.warn(f"No tensor inputs to compute gradients for sample input: {sample_input.summary(True)}")
                continue
            # Select grad positions: if self.grad_position is provided, keep only those overlapping
            # with grad-capable tensor indices; otherwise use all grad-capable tensor indices.
            selected_grad_position = tensor_indices
            if self.grad_position is not None:
                selected_grad_position = tuple(i for i in self.grad_position if i in tensor_indices)
                if not selected_grad_position:
                    grads.append(tuple())
                    warnings.warn("The input on the grad position you specified are all undifferentiable.")
                    continue
            grad_func = grad_func or ms.grad(self.customize_op_func, grad_position=selected_grad_position)
            # Use "ms_backend" to keep forward/backward consistent in kbk mode.
            if ms.context.get_context('mode') == ms.GRAPH_MODE:
                grad_func = ms.jit(grad_func, backend="ms_backend")
            grad_outi = grad_func(*args_no_dout)
            if not isinstance(grad_outi, (tuple, list)):
                # Single grad output: keep only if the first input is tensor
                grad_outi = (grad_outi,)
            grads.append(grad_outi)

        _pynative_executor.sync()
        return grads

    def grad_pytorch_impl(
            self,
            *args,
            **kwargs
    ):
        """Compute gradients with the PyTorch reference implementation.

        Computes gradients for all tensor inputs among (op_input, *op_args).

        Returns:
            list[tuple]: Per-sample tuple of gradients matching tensor inputs order.
        """
        #TODO: use customized dout instead of ones_like when ms.grad supports dout input
        #torch_douts = self._generate_random_dout(return_torch_douts=True)

        torch_fn = self.ref
        grads = []

        def _torch_dtype_supports_grad(t: torch.Tensor) -> bool:
            return torch.is_floating_point(t) or torch.is_complex(t)

        def _flatten_tensors(obj):
            """Flatten grad-capable torch.Tensor leaves in a container, preserving order."""
            if isinstance(obj, torch.Tensor):
                return [obj] if _torch_dtype_supports_grad(obj) else []
            elif isinstance(obj, (tuple, list)):
                return [t for v in obj for t in _flatten_tensors(v)]
            else:
                return []

        for sample_input in self._sample_inputs:
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.astorch(convert_half_to_float=self._convert_half_to_float,
                            convert_scalar_to_tensor=getattr(self.op_info, 'convert_ref_scalar_to_tensor',False))
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs
            # handle op_input
            tensor_inputs = []
            if isinstance(op_input, torch.Tensor) and _torch_dtype_supports_grad(op_input):
                if self.grad_position is None or 0 in self.grad_position:
                    op_input.requires_grad = True
                tensor_inputs.append(('input', op_input))
            elif isinstance(op_input, (tuple, list)):
                if not self.flatten_inputs_for_backward:
                    warnings.warn(
                        "op_input is a sequence, but torch does not support backward for "
                        "sequence inputs; please consider setting "
                        "flatten_inputs_for_backward to True"
                    )
                    continue
                # Some ops (e.g. cat) take a sequence of tensors as the first argument.
                for t in _flatten_tensors(op_input):
                    if self.grad_position is None or 0 in self.grad_position:
                        t.requires_grad = True
                    tensor_inputs.append(('input', t))
            # handle op_args
            arg_tensors = []
            for i, arg in enumerate(op_args):
                if isinstance(arg, torch.Tensor) and _torch_dtype_supports_grad(arg):
                    if self.grad_position is None or i + 1 in self.grad_position:
                        arg.requires_grad = True
                    arg_tensors.append(arg)
                elif isinstance(arg, (tuple, list)):
                    for t in _flatten_tensors(arg):
                        if self.grad_position is None or i + 1 in self.grad_position:
                            t.requires_grad = True
                        arg_tensors.append(t)
            tensor_inputs.extend(('arg', t) for t in arg_tensors)

            outi = torch_fn(op_input, *op_args, **op_kwargs)
            # If no grad-capable inputs, skip backward to avoid autograd errors
            if not tensor_inputs:
                grads.append(tuple())
                continue
            # Support multi-output backward with matching grad structure
            # dout_i = torch_douts[idx]
            if isinstance(outi, (tuple, list)):
                grad_list = [torch.ones_like(o) for o in outi]
                tensor_require_grad = [t for t in outi if t.requires_grad]
                grad_list_filtered = [g for t, g in zip(outi, grad_list) if t.requires_grad]
                torch.autograd.backward(tensor_require_grad, grad_tensors=grad_list_filtered)
            else:
                outi_grad = torch.ones_like(outi)
                outi.backward(gradient=outi_grad)

            grad_tuple = []
            for _, tin in tensor_inputs:
                grad = tin.grad
                if grad is not None:
                    grad_tuple.append(grad.detach())
            grads.append(tuple(grad_tuple))

        return grads

    def grad_tensorflow_impl(
            self,
            *args,
            **kwargs
    ):
        """Compute gradients with the TensorFlow reference implementation."""
        raise NotImplementedError

    def grad_numpy_impl(
            self,
            *args,
            **kwargs
    ):
        """Compute gradients with the NumPy reference implementation."""
        raise NotImplementedError


    def forward_mindspore_dynamic_shape_impl(
            self,
            *args,
            **kwargs
    ):
        """Run forward with MindSpore for dynamic-shape execution.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Outputs per dynamic-shape sample.
        """

        compile_inputs = self._dynamic_inputs.op_compile_input.convert_to_args().op_args
        _code = self.customize_op_func.__code__
        arg_names = _code.co_varnames[:_code.co_argcount]
        dyn_kwargs = {name: val for name, val in zip(arg_names, compile_inputs) if is_op_input_dynamic(val)}

        dyn_op_func = ms.enable_dynamic(**dyn_kwargs)(ms.jit(self.customize_op_func, backend="ms_backend"))
        out = []
        for running_input in self._dynamic_inputs.op_running_inputs:
            if self._inplace_op:
                running_input = running_input.copy()

            running_input = running_input.convert_to_args()
            outi = dyn_op_func(*running_input.op_args)
            out.append(outi)

        _pynative_executor.sync()
        return out

    def forward_pytorch_dynamic_shape_impl(
            self,
            *args,
            **kwargs
    ):
        """Run forward with PyTorch for dynamic-shape execution.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Outputs per dynamic-shape sample.
        """
        torch_fn = self.ref
        out = []

        for running_input in self._dynamic_inputs.op_running_inputs:
            if self._inplace_op:
                running_input = running_input.copy()
            running_input = running_input.astorch(convert_half_to_float=self._convert_half_to_float,
                                                  convert_extra_uint=self._convert_extra_uint,
                            convert_scalar_to_tensor=getattr(self.op_info, 'convert_ref_scalar_to_tensor',False))
            op_input, op_args, op_kwargs = running_input.op_input, running_input.op_args, running_input.op_kwargs
            outi = torch_fn(op_input, *op_args, **op_kwargs)
            out.append(outi)

        return out

    def grad_mindspore_dynamic_shape_impl(
            self,
            *args,
            **kwargs
    ):
        """Compute gradients with MindSpore for dynamic-shape execution.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Gradients per dynamic-shape sample.
        """
        def _ms_tensor_supports_grad(t):
            return isinstance(t, ms.Tensor) and (t.dtype.is_floating_point or t.dtype.is_complex)

        grads = []
        compile_inputs = self._dynamic_inputs.op_compile_input.convert_to_args(
            flatten_op_input=self.flatten_inputs_for_backward
        ).op_args
        tensor_indices = tuple(i for i, v in enumerate(compile_inputs) if _ms_tensor_supports_grad(v))
        if not tensor_indices:
            grads.append(tuple())
            warnings.warn("No tensor inputs to compute gradients for compile input")
            return grads
        # Select grad positions: if self.grad_position is provided, keep only those overlapping
        # with grad-capable tensor indices; otherwise use all grad-capable tensor indices.
        selected_grad_position = tensor_indices
        if self.grad_position is not None:
            selected_grad_position = tuple(i for i in self.grad_position if i in tensor_indices)
            if not selected_grad_position:
                grads.append(tuple())
                warnings.warn("The input on the grad position you specified are all undifferentiable.")
                return grads

        grad_net = self._op_grad_cell(self.customize_op_func, grad_position=selected_grad_position)
        # dynamic inputs for jit must be all keyword arguments, so use set_inputs to set the dynamic inputs.
        grad_net.set_inputs(*compile_inputs)

        for running_input in self._dynamic_inputs.op_running_inputs:
            if self._inplace_op:
                running_input = running_input.copy()
            args_no_dout = running_input.convert_to_args(flatten_op_input=self.flatten_inputs_for_backward).op_args

            # After convert_to_args, op_input, op_args and op_kwargs are all in op_args now.
            grad_outi = grad_net(*args_no_dout)
            if not isinstance(grad_outi, (tuple, list)):
                # Single grad output: keep only if the first input is tensor
                grad_outi = (grad_outi,)
            grads.append(grad_outi)

        _pynative_executor.sync()
        return grads


    def grad_pytorch_dynamic_shape_impl(
            self,
            *args,
            **kwargs
    ):
        """Compute gradients with PyTorch for dynamic-shape execution.

        Computes gradients for all tensor inputs among (op_input, *op_args).

        Returns:
            list[tuple]: Per-sample tuple of gradients matching tensor inputs order.
        """
        torch_fn = self.ref
        grads = []

        def _torch_dtype_supports_grad(t: torch.Tensor) -> bool:
            return torch.is_floating_point(t) or torch.is_complex(t)

        def _flatten_tensors(obj):
            """Flatten grad-capable torch.Tensor leaves in a container, preserving order."""
            if isinstance(obj, torch.Tensor):
                return [obj] if _torch_dtype_supports_grad(obj) else []
            if isinstance(obj, (tuple, list)):
                return [t for v in obj for t in _flatten_tensors(v)]
            return []

        for running_input in self._dynamic_inputs.op_running_inputs:
            if self._inplace_op:
                running_input = running_input.copy()
            running_input = running_input.astorch(convert_half_to_float=self._convert_half_to_float,
                         convert_scalar_to_tensor=getattr(self.op_info, 'convert_ref_scalar_to_tensor',False))
            op_input, op_args, op_kwargs = running_input.op_input, running_input.op_args, running_input.op_kwargs

            tensor_inputs = []
            if isinstance(op_input, torch.Tensor) and _torch_dtype_supports_grad(op_input):
                if self.grad_position is None or 0 in self.grad_position:
                    op_input.requires_grad = True
                tensor_inputs.append(('input', op_input))
            elif isinstance(op_input, (tuple, list)):
                # Some ops (e.g. cat) take a sequence of tensors as the first argument.
                for t in _flatten_tensors(op_input):
                    if self.grad_position is None or 0 in self.grad_position:
                        t.requires_grad = True
                    tensor_inputs.append(('input', t))
            arg_tensors = []
            for i, arg in enumerate(op_args):
                if isinstance(arg, torch.Tensor) and _torch_dtype_supports_grad(arg):
                    if self.grad_position is None or i + 1 in self.grad_position:
                        arg.requires_grad = True
                    arg_tensors.append(arg)
                elif isinstance(arg, (tuple, list)):
                    for t in _flatten_tensors(arg):
                        if self.grad_position is None or i + 1 in self.grad_position:
                            t.requires_grad = True
                        arg_tensors.append(t)
            tensor_inputs.extend(('arg', t) for t in arg_tensors)

            outi = torch_fn(op_input, *op_args, **op_kwargs)
            if not tensor_inputs:
                grads.append(tuple())
                continue
            # For dynamic, use ones_like grads; handle multi-output
            if isinstance(outi, (tuple, list)):
                grad_list = [torch.ones_like(o) for o in outi]
                torch.autograd.backward(list(outi), grad_tensors=grad_list)
            else:
                outi_grad = torch.ones_like(outi)
                outi.backward(gradient=outi_grad)

            grad_tuple = []
            for _, tin in tensor_inputs:
                grad_tuple.append(tin.grad.detach())
            grads.append(tuple(grad_tuple))

        return grads

    def compare_with_torch(
            self,
            *,
            sample_inputs: Union[List[OpSampleInput], OpSampleInput],
            grad_cmp: Optional[bool] = False,
            ksize: Optional[int] = 1, # ksize for elementwise op, set other value if you want
    ):
        """Compare MindSpore outputs/gradients with PyTorch on static shapes.

        Args:
            sample_inputs: Single or list of sample inputs.
            grad_cmp: When True and differentiable, compare gradients.
            ksize: Optional kernel size hint for comparison helpers.
        """
        self._sample_inputs = sample_inputs if isinstance(sample_inputs, list) else [sample_inputs]

        if grad_cmp and self.op_info.is_differentiable:
            ms_out = self.grad_mindspore_impl()
            pt_out = self.grad_pytorch_impl()
        else:
            ms_out = self.forward_mindspore_impl()
            pt_out = self.forward_pytorch_impl()

        # Some ops are inherently non-deterministic (e.g., random sampling). For those ops, we still
        # execute forward to catch runtime errors, but skip output comparison.
        if not grad_cmp and self.op_info.skip_forward_comparison:
            return

        for sample_input, ms_outi, pt_outi in zip(self._sample_inputs, ms_out, pt_out):
            sample_loss = getattr(sample_input, "loss", None)
            if isinstance(ms_outi, (tuple, list)) and isinstance(pt_outi, (tuple, list)):
                # The output of the op maybe a tuple or list for some multi-output ops.
                for ms_outi_tensor, pt_outi_tensor in zip(ms_outi, pt_outi):
                    self.assert_equal(
                        ms_outi_tensor,
                        pt_outi_tensor,
                        rtol=sample_loss,
                        atol=sample_loss,
                        compare_method=self._compare_method,
                        grad_cmp=grad_cmp,
                        ksize=ksize,
                        op_type=OpTypes.COMPUTE_FLOAT
                    )
            else:
                self.assert_equal(
                    ms_outi,
                    pt_outi,
                    rtol=sample_loss,
                    atol=sample_loss,
                    compare_method=self._compare_method,
                    grad_cmp=grad_cmp,
                    ksize=ksize,
                    op_type=OpTypes.COMPUTE_FLOAT
                )

    def compare_with_torch_dynamic(
            self,
            *,
            op_dynamic_inputs: OpDynamicInput,
            grad_cmp: Optional[bool] = False,
            ksize: Optional[int] = 1, # ksize for elementwise op, set other value if you want
    ):
        """Compare MindSpore with PyTorch under dynamic-shape execution.

        Args:
            op_dynamic_inputs: OpDynamicInput object.
            grad_cmp: When True and differentiable, compare gradients.
            ksize: Optional kernel size hint for comparison helpers.
        """
        self._dynamic_inputs = op_dynamic_inputs

        if grad_cmp and self.op_info.is_differentiable:
            ms_out = self.grad_mindspore_dynamic_shape_impl()
            pt_out = self.grad_pytorch_dynamic_shape_impl()
        else:
            ms_out = self.forward_mindspore_dynamic_shape_impl()
            pt_out = self.forward_pytorch_dynamic_shape_impl()

        # Still execute forward, but skip output comparison for non-deterministic ops.
        if not grad_cmp and self.op_info.skip_forward_comparison:
            return

        for running_input, ms_outi, pt_outi in zip(self._dynamic_inputs.op_running_inputs, ms_out, pt_out):
            sample_loss = getattr(running_input, "loss", None)
            if isinstance(ms_outi, (tuple, list)) and isinstance(pt_outi, (tuple, list)):
                # The output of the op maybe a tuple or list for some multi-output ops.
                for ms_outi_tensor, pt_outi_tensor in zip(ms_outi, pt_outi):
                    self.assert_equal(
                        ms_outi_tensor,
                        pt_outi_tensor,
                        rtol=sample_loss,
                        atol=sample_loss,
                        grad_cmp=grad_cmp,
                    )
            else:
                self.assert_equal(ms_outi, pt_outi, rtol=sample_loss, atol=sample_loss, grad_cmp=grad_cmp)

    def test_op_reference(
            self,
            *,
            grad_cmp: bool = False,
    ):
        """Run reference parity tests against Benchmark for all supported dtypes.

        Args:
            grad_cmp: When True, restrict to floating dtypes and compare first-order gradients.
        """
        if self.op_basic_reference_inputs_func is None:
            print(f"\nop_name: {self.op_name} has no op_basic_reference_inputs_func, skip test_op_reference.")
            return

        try:
            print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, test_op_reference...")
            if grad_cmp:
                self.supported_dtypes = tuple(d for d in self.supported_dtypes if d.is_floating_point)
            for dtype in self.supported_dtypes:
                if grad_cmp:
                    for sample_input in self.op_basic_reference_inputs_func(self.op_info, dtype, device=self._device):
                        self.compare_with_torch(sample_inputs=sample_input, grad_cmp=True)
                else:
                    for sample_input in self.op_basic_reference_inputs_func(self.op_info, dtype, device=self._device):
                        self.compare_with_torch(sample_inputs=sample_input)
                    if self.op_extra_reference_inputs_func is not None:
                        for sample_input in self.op_extra_reference_inputs_func(
                                self.op_info,
                                dtype,
                                device=self._device,
                        ):
                            self.compare_with_torch(sample_inputs=sample_input)
        except Exception as e:
            error_msg = (f"\ntest_op_reference failed:"
                        f"\nop_name: {self.op_name}"
                        f"\nmode: {self._context_mode}"
                        f"\ndtype: {dtype}")
            if 'sample_input' in locals():
                error_msg += f"\n{sample_input.summary(True)}"
            print(error_msg)
            raise e

    def test_op_dynamic(
            self,
            *,
            grad_cmp: bool = False,
            only_dynamic_shape: bool = False,
            only_dynamic_rank: bool = False,
            dtype = ms.float32,
    ):
        """Run dynamic-shape tests against Benchmark.

        Args:
            grad_cmp: When True, also compare first-order gradients.
            only_dynamic_shape: If True, only run dynamic-shape cases (fixed rank).
            only_dynamic_rank: If True, only run dynamic-rank cases (shape varies in rank).
            dtype: Dtype used by dynamic input generator; default float32.
        """
        if self.op_info.op_dynamic_inputs_func is None:
            print(f"\nop_name: {self.op_name} has no op_dynamic_inputs_func, skip test_op_dynamic.")
            return

        if self._device == 'ascend':
            ascend_name = MSContext.get_instance().get_ascend_soc_version()
            if ascend_name == 'ascend910' and not self.op_info.dtypes_ascend:
                warnings.warn(f"op_name: {self.op_name} has no dtypes_ascend, skip test_op_dynamic.")
                return
            if ascend_name == 'ascend910b' and not self.op_info.dtypes_ascend910b:
                warnings.warn(f"op_name: {self.op_name} has no dtypes_ascend910b, skip test_op_dynamic.")
                return
        if self._device == 'cpu' and not self.op_info.dtypes_cpu:
            warnings.warn(f"op_name: {self.op_name} has no dtypes_cpu, skip test_op_dynamic.")
            return
        if self._device == 'gpu' and not self.op_info.dtypes_gpu:
            warnings.warn(f"op_name: {self.op_name} has no dtypes_gpu, skip test_op_dynamic.")
            return

        try:
            print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, test_op_dynamic...")
            if grad_cmp:
                self.supported_dtypes = tuple(d for d in self.supported_dtypes if d.is_floating_point)
                if not self.supported_dtypes:
                    return
                if ms.float32 in self.supported_dtypes:
                    dtype = ms.float32
                else:
                    dtype = self.supported_dtypes[0]
            for op_dynamic_input in self.op_info.op_dynamic_inputs_func(
                    self.op_info,
                    dtype=dtype,
                    device=self._device,
                    only_dynamic_shape=only_dynamic_shape,
                    only_dynamic_rank=only_dynamic_rank):
                if grad_cmp:
                    self.compare_with_torch_dynamic(op_dynamic_inputs=op_dynamic_input, grad_cmp=True)
                else:
                    self.compare_with_torch_dynamic(op_dynamic_inputs=op_dynamic_input)
        except Exception as e:
            error_msg = (f"\ntest_op_dynamic failed:"
                        f"\nop_name: {self.op_name}"
                        f"\nmode: {self._context_mode}")
            if 'op_dynamic_input' in locals():
                error_msg += f"\n{op_dynamic_input.op_compile_input.summary()}"
            print(error_msg)
            raise e

    def forward_cmp(
            self,
            *args,
            rtol=None,
            atol=None,
            benchmark='torch',
            **kwargs,
    ):
        """Compare MindSpore forward results with a reference implementation.

        Args:
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            benchmark: 'torch' | 'numpy'.
        """
        ms_out = self.forward_mindspore_impl()
        if benchmark == 'torch':
            pt_out = self.forward_pytorch_impl()
        elif benchmark == 'numpy':
            pt_out = self.forward_numpy_impl()
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}, expected: 'torch', 'numpy'.")

        if self.op_info.skip_forward_comparison:
            return

        for ms_outi, pt_outi in zip(ms_out, pt_out):
            if isinstance(ms_outi, (tuple, list)) and isinstance(pt_outi, (tuple, list)):
                # The output of the op maybe a tuple or list for some multi-output ops.
                for ms_outi_tensor, pt_outi_tensor in zip(ms_outi, pt_outi):
                    self.assert_equal(ms_outi_tensor, pt_outi_tensor, rtol, atol)
            else:
                self.assert_equal(ms_outi, pt_outi, rtol, atol)

    def grad_cmp(
            self,
            *args,
            rtol=None,
            atol=None,
            benchmark='torch',
            **kwargs,
    ):
        """Compare MindSpore gradients with a reference implementation.

        Args:
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            benchmark: 'torch' | 'numpy'.
        """
        ms_grads = self.grad_mindspore_impl()
        if benchmark == 'torch':
            pt_grads = self.grad_pytorch_impl()
        elif benchmark == 'numpy':
            pt_grads = self.grad_numpy_impl()
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}, expected: 'torch', 'numpy'.")

        for ms_gradi, pt_gradi in zip(ms_grads, pt_grads):
            if isinstance(ms_gradi, (tuple, list)) and isinstance(pt_gradi, (tuple, list)):
                # The gradient of the op maybe a tuple or list for some multi-tensor input ops.
                for ms_gradi_tensor, pt_gradi_tensor in zip(ms_gradi, pt_gradi):
                    self.assert_equal(ms_gradi_tensor, pt_gradi_tensor, rtol, atol, grad_cmp=True)
            else:
                self.assert_equal(ms_gradi, pt_gradi, rtol, atol, grad_cmp=True)

    def forward_dynamic_shape_cmp(
            self,
            *args,
            rtol=None,
            atol=None,
            benchmark='torch',
            **kwargs,
    ):
        """Compare forward results under dynamic-shape execution.

        Args:
            op_dynamic_inputs: OpDynamicInput object.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            benchmark: 'torch'.
        """
        if self._context_mode == 'pynative':
            raise RuntimeError("Dynamic shape comparison is not supported in pynative mode.")

        ms_outs = self.forward_mindspore_dynamic_shape_impl()
        if benchmark == 'torch':
            pt_outs = self.forward_pytorch_dynamic_shape_impl()
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}, expected: 'torch'.")

        if self.op_info.skip_forward_comparison:
            return

        for ms_outi, pt_outi in zip(ms_outs, pt_outs):
            if isinstance(ms_outi, (tuple, list)) and isinstance(pt_outi, (tuple, list)):
                # The output of the op with dynamic shape maybe a tuple or list for some multi-output ops.
                for ms_outi_tensor, pt_outi_tensor in zip(ms_outi, pt_outi):
                    self.assert_equal(ms_outi_tensor, pt_outi_tensor, rtol, atol)
            else:
                self.assert_equal(ms_outi, pt_outi, rtol, atol)

    def grad_dynamic_shape_cmp(
            self,
            *args,
            rtol=None,
            atol=None,
            benchmark='torch',
            **kwargs,
    ):
        """Compare gradients under dynamic-shape execution.

        Args:
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            benchmark: 'torch'.
        """
        if self._context_mode == 'pynative':
            raise RuntimeError("Dynamic shape comparison is not supported in pynative mode.")

        ms_grads = self.grad_mindspore_dynamic_shape_impl()
        if benchmark == 'torch':
            pt_grads = self.grad_pytorch_dynamic_shape_impl()
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}, expected: 'torch'.")
        for ms_gradi, pt_gradi in zip(ms_grads, pt_grads):
            if isinstance(ms_gradi, (tuple, list)) and isinstance(pt_gradi, (tuple, list)):
                # The gradient of the op maybe a tuple or list for some multi-tensor input ops.
                for ms_gradi_tensor, pt_gradi_tensor in zip(ms_gradi, pt_gradi):
                    self.assert_equal(ms_gradi_tensor, pt_gradi_tensor, rtol, atol, grad_cmp=True)
            else:
                self.assert_equal(ms_gradi, pt_gradi, rtol, atol, grad_cmp=True)

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
- OpSampleInput: a lightweight container for op inputs/args/kwargs with
  convenient transformations (copy/asnumpy/astorch/discontiguous).
- Tensor helpers for discontiguity, host sync, and tensor creation.
"""
# pylint: disable=R1705
import math
import torch
import numpy as np
import mindspore as ms
from mindspore.common.api import _pynative_executor
from mindspore.common.dtype import _dtype_to_nptype
from mindspore._c_expression import MSContext
from tests.st.ops.share._op_info.op_common import dtypes_common
from typing import Optional, Tuple, Union, Dict, List, Any

class OpSampleInput:
    """Container of a single operation invocation sample.

    Attributes:
        op_input: The first positional input (commonly a Tensor).
        op_args: Extra positional arguments of the operator.
        op_kwargs: Keyword arguments of the operator.
        sample_name: A short name for identification in logs.
    """

    __slots__ = [
        "op_input",
        "op_args",
        "op_kwargs",
        "sample_name",
        "loss",
    ]

    def __init__(
            self,
            op_input,
            op_args: Optional[tuple] = tuple(),
            op_kwargs: Optional[dict] = None,
            sample_name: Optional[str] = None,
            loss: Optional[float] = None,
    ):
        self.op_input = op_input
        self.op_args = op_args
        self.op_kwargs = op_kwargs if op_kwargs is not None else {}
        self.sample_name = sample_name if sample_name is not None else "UnknownSample"
        self.loss = loss

    def transform(self, fn, method_name, convert_scalar_to_tensor = False):
        """Apply a transformation recursively to op_input/op_args/op_kwargs.

        Args:
            fn: A callable used to transform each leaf element.
            method_name: Suffix appended to `sample_name` for traceability.
            convert_scalar_to_tensor: If True, convert scalar to Tensor.
        Returns:
            A new OpSampleInput with transformed fields.
        """
        def _transform(x):
            def _transform_to(x):
                return fn(x)

            if getattr(x, '__ms_mutable__', False):
                return _transform_to(x.__ms_origin_object__)
            elif isinstance(x, ms.Tensor):
                return _transform_to(x)
            elif x in ms.dtype.all_types:
                return _transform_to(x)
            elif isinstance(x, list):
                return list(map(_transform, x))
            elif isinstance(x, tuple):
                return tuple(map(_transform, x))
            elif isinstance(x, dict):
                return {k: _transform(v) for k, v in x.items()}
            else:
                if convert_scalar_to_tensor:
                    return _transform_to(x)
                else:
                    return x

        transformed_op_input, transformed_op_args, transformed_op_kwargs = (
            _transform(self.op_input),
            _transform(self.op_args),
            _transform(self.op_kwargs),
        )

        return OpSampleInput(
            transformed_op_input,
            op_args=transformed_op_args,
            op_kwargs=transformed_op_kwargs,
            sample_name=self.sample_name + "_transformed_" + method_name,
            loss=self.loss,
        )

    def convert_to_args(self, flatten_op_input=False, append_dout=None):
        """Flatten input/args/kwargs (and optional dout) into a single args tuple.

        Args:
            append_dout: Optional extra output gradients (sens) to append as a single
                positional argument. For multi-output ops, pass a tuple of dout
                tensors and it will be appended as one argument without expansion.

        Returns:
            A new OpSampleInput whose `op_args` contains all flattened arguments,
            and `op_input`/`op_kwargs` are cleared.
        """
        def _to_args_list(x):
            if isinstance(x, dict):
                return list(x.values())
            elif isinstance(x, (list, tuple)):
                return list(x)
            else:
                return [x]

        op_args = []
        op_args.extend([self.op_input] if not flatten_op_input else _to_args_list(self.op_input))
        op_args.extend(_to_args_list(self.op_args))
        op_args.extend(_to_args_list(self.op_kwargs))
        if append_dout is not None:
            # Append sens as a single argument (do NOT expand lists/tuples),
            # so that multi-output dout can be passed correctly.
            op_args.append(append_dout)

        return OpSampleInput(
            op_input=None,
            op_args=tuple(op_args),
            op_kwargs={},
            sample_name=self.sample_name + ("_to_args_with_dout" if append_dout is not None else "_to_args"),
            loss=self.loss,
        )

    def copy(self):
        """Deep-ish copy of Tensor-like content.

        Ensures ms.Tensor elements are copied while preserving structure.
        """
        def _copy(x):
            if isinstance(x, ms.Tensor):
                return x.copy()
            elif x in ms.dtype.all_types:
                return x
            elif isinstance(x, list):
                return list(map(_copy, x))
            elif isinstance(x, tuple):
                return tuple(map(_copy, x))
            elif isinstance(x, dict):
                return {k: _copy(v) for k, v in x.items()}
            else:
                return x
        return self.transform(_copy, 'copy')

    def asnumpy(self):
        """Convert all Tensor leaves to numpy arrays (values copied)."""
        def _asnumpy(x):
            if isinstance(x, ms.Tensor):
                return ms_asnumpy(x).copy()
            elif x in ms.dtype.all_types:
                return _dtype_to_nptype(x)

            return x

        return self.transform(_asnumpy, 'asnumpy')

    def astorch(
            self,
            *,
            convert_half_to_float: Optional[bool] = False,
            convert_extra_uint: Optional[bool] = False,
            convert_scalar_to_tensor: Optional[bool] = False,
    ):
        """Convert MindSpore tensors and dtypes to PyTorch counterparts.

        Args:
            convert_half_to_float: Cast float16 to float32 for reference backends.
            convert_extra_uint: Convert extra uint dtypes to supported types.
            convert_scalar_to_tensor : Convert scalar dtypes to supported types.
        Returns:
            A new OpSampleInput converted for PyTorch execution.
        """
        def _dtype_to_torch_dtype(msdtype):
            msdtype_to_torch_dtype_dict = {
                ms.bool_: torch.bool,
                ms.int8: torch.int8,
                ms.int16: torch.int16,
                ms.int32: torch.int32,
                ms.int64: torch.int64,
                ms.uint8: torch.uint8,
                ms.float16: torch.float16,
                ms.float32: torch.float32,
                ms.float64: torch.float64,
                ms.complex64: torch.complex64,
                ms.complex128: torch.complex128,
                ms.bfloat16: torch.bfloat16,
            }
            return msdtype_to_torch_dtype_dict[msdtype]

        def _astorch(x):
            if isinstance(x, ms.Tensor):
                np_arr = ms_asnumpy(x,
                                    convert_half_to_float=convert_half_to_float,
                                    convert_extra_uint=convert_extra_uint)
                return torch.tensor(np_arr, dtype=torch.bfloat16) if x.dtype == ms.bfloat16 else torch.tensor(np_arr)
            elif convert_scalar_to_tensor and type(x) in dtypes_common:
                return torch.tensor(x)
            elif x in ms.dtype.all_types:
                return _dtype_to_torch_dtype(x)

            return x

        return self.transform(_astorch, 'astorch', convert_scalar_to_tensor = convert_scalar_to_tensor)

    def discontiguous(self):
        """Make all Tensor leaves discontiguous in memory when possible."""
        def _discontiguous(x):
            if isinstance(x, ms.Tensor):
                return _tensor_to_discontiguous(x)
            elif isinstance(x, ms.dtype):
                return x

            return x

        if 'transformed_astorch' in self.sample_name:
            raise RuntimeError("OpSampleInput only supports discontiguous method with mindspore.Tensor now.")

        return self.transform(_discontiguous, 'discontiguous')

    def summary(self, values=False):
        """Human-readable summary for debugging.

        Args:
            values: Whether to include stats like mean/max/min for arrays.

        Returns:
            A string summary of this sample input.
        """
        def _tensor_summary(x):
            if isinstance(x, (ms.Tensor, torch.Tensor, np.ndarray)):
                sum_info = f"{type(x).__name__}(shape={x.shape}, dtype={x.dtype}"
                if values:
                    np_x = tensor_to_numpy(x)
                    # NOTE: For empty tensors, numpy reduction like max/min raises:
                    # "zero-size array to reduction operation maximum which has no identity".
                    # Keep summary robust so debugging/logging never masks the real failure.
                    if np_x.size == 0:
                        sum_info += ", empty=True"
                    else:
                        x_mean = np_x.mean()
                        x_max = np_x.max()
                        x_min = np_x.min()
                        sum_info += f", mean={x_mean}, max={x_max}, min={x_min}"
                return sum_info + ")"
            elif isinstance(x, list):
                return "list[" + ", ".join(map(_tensor_summary, x)) + "]"
            elif isinstance(x, tuple):
                return "tuple(" + ", ".join(map(_tensor_summary, x)) + ")"
            elif isinstance(x, dict):
                return "dict(" + ", ".join(f"{k}: {_tensor_summary(v)}" for k, v in x.items()) + ")"
            else:
                return f"{type(x).__name__}({x})"

        return self.__repr__(_tensor_summary)

    def __repr__(self, print_func=lambda x: x):
        return "OpSampleInput(\n" + \
               f"op_input={print_func(self.op_input)},\n" + \
               f"op_args={print_func(self.op_args)},\n" + \
               f"op_kwargs={print_func(self.op_kwargs)},\n" + \
               f"sample_name={print_func(self.sample_name)},\n" + \
               f"loss={print_func(self.loss)})"


class OpErrorInput:
    '''
    Container of a single error input sample.
    Attributes:
        op_sample_input: The sample input that caused the error.
        op_error_type: The type of error.
        op_error_info: The info of error.
    '''
    __slots__ = [
        "op_sample_input",
        "op_error_type",
        "op_error_info",
    ]

    def __init__(
        self,
        op_sample_input: OpSampleInput,
        op_error_type,
        op_error_info,
    ):
        self.op_sample_input = op_sample_input
        self.op_error_type = op_error_type
        self.op_error_info = op_error_info


class OpDynamicInput:
    '''
    Container of a couple of input sample.
    Attributes:
        op_compile_input: The sample input for compiling.
        op_running_inputs: The sample input for running.
    '''
    __slots__ = [
        "op_compile_input",
        "op_running_inputs",
    ]

    def __init__(
        self,
        op_compile_input: OpSampleInput,
        op_running_inputs: Tuple[OpSampleInput],
    ):
        self.op_compile_input = op_compile_input
        self.op_running_inputs = op_running_inputs


def _tensor_to_discontiguous(x):
    """Return a view that is not contiguous in memory when feasible.

    For non-trivial tensors, materialize a new last dimension to break
    contiguity and return the second slice view.
    """
    if not x.is_contiguous():
        return x

    if x.numel() == 0 or x.numel() == 1:
        return x

    empty_tensor = x.new_empty(x.shape + (2,))
    # set magic number for unusable memory.
    if x.dtype == ms.bool_:
        empty_tensor[..., 0] = True
    elif not x.is_floating_point() and not x.is_complex():
        empty_tensor[..., 0] = 77
    else:
        empty_tensor[..., 0] = math.nan
    empty_tensor[..., 1] = x.copy()
    result = empty_tensor[..., 1]

    assert not result.is_contiguous()
    return result


def tensor_to_numpy(tensor):
    """
    Convert a tensor to numpy array.
    Args:
        tensor: Tensor to convert.
    Returns:
        Numpy ndarray on host.
    """
    if isinstance(tensor, torch.Tensor):
        return torch_asnumpy(tensor)
    elif isinstance(tensor, ms.Tensor):
        return ms_asnumpy(tensor)
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")


def torch_asnumpy(tensor, convert_half_to_float=False, convert_extra_uint=False):
    """
    Convert a PyTorch tensor to numpy array with optional casts.
    Args:
        tensor: PyTorch tensor to convert.
        convert_half_to_float: If True, cast float16 to float32 before copying.
        convert_extra_uint: If True, cast uint16/32/64 to int64 for compatibility.
    Returns:
        Numpy ndarray on host.
    """
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    if convert_half_to_float and tensor.dtype == torch.float16:
        tensor = tensor.float()
    if convert_extra_uint and tensor.dtype in (torch.uint16, torch.uint32, torch.uint64):
        tensor = tensor.int()
    return tensor.cpu().detach().numpy()


def ms_asnumpy(tensor, convert_half_to_float=False, convert_extra_uint=False):
    """Convert a MindSpore tensor to numpy array with optional casts.

    Args:
        tensor: MindSpore Tensor to convert.
        convert_half_to_float: If True, cast float16 to float32 before copying.
        convert_extra_uint: If True, cast uint16/32/64 to int64 for compatibility.

    Returns:
        Numpy ndarray on host.
    """
    def _sync_host(tensor):
        try:
            host_tensor = tensor.to('cpu')
            _pynative_executor.sync()
            return host_tensor
        except Exception:  # pylint: disable=W0703
            return tensor

    if not isinstance(tensor, ms.Tensor):
        raise ValueError(f"tensor must be a ms.Tensor, but got {type(tensor)}")

    if tensor.dtype == ms.bfloat16:
        return _sync_host(tensor).float().asnumpy()
    if convert_half_to_float and tensor.dtype == ms.float16:
        return _sync_host(tensor).float().asnumpy()
    if convert_extra_uint and tensor.dtype in (ms.uint16, ms.uint32, ms.uint64):
        return _sync_host(tensor).asnumpy().astype(np.int64)
    return _sync_host(tensor).asnumpy()


def make_tensor(
        shape: tuple[int, ...],
        dtype: ms.dtype,
        low: Optional[float] = None,
        high: Optional[float] = None,
        exclude_value: Optional[tuple[Any,...]] = None,
        *,
        device: Optional[str] = None,
        discontiguous: Optional[bool] = False,
        random_seed: Optional[int] = None,
        random_method: Optional[str] = None,
):
    """Create a MindSpore Tensor with random contents for testing.
       Args:
           exclude_value: exclude values from for random tensor creation
       """
    def _generate_ndarray(shape, dtype, low, high, random_method, exclude_value = None):
        """Create a MindSpore Tensor with random contents for testing.
            Args:
              exclude_value: exclude values from for random tensor creation
            """
        def _generate_ndarray_by_random_method(random_method, shape, dtype, low, high):
            if random_method == 'randn':
                ndarray = np.clip(np.random.randn(*shape), low, high)
            elif random_method == 'randint':
                ndarray = np.random.randint(math.ceil(low), high, size=shape)
            elif random_method == 'uniform':
                ndarray = np.random.uniform(low, high, size=shape)
            else:
                raise ValueError(f"Invalid random method: {random_method}")

            if isinstance(ndarray, np.ndarray):
                ndarray = ndarray.astype(dtype)
            return ndarray

        dtype_to_np_dtype_dict = {
            ms.bool_: (np.bool_, 'randint', 0, 2),
            ms.int8: (np.int8, 'randint', -9, 10),
            ms.int16: (np.int16, 'randint', -9, 10),
            ms.int32: (np.int32, 'randint', -9, 10),
            ms.int64: (np.int64, 'randint', -9, 10),
            ms.uint8: (np.uint8, 'randint', 0, 10),
            ms.uint16: (np.uint16, 'randint', 0, 10),
            ms.uint32: (np.uint32, 'randint', 0, 10),
            ms.uint64: (np.uint64, 'randint', 0, 10),
            ms.float16: (np.float16, 'uniform', -9, 9),
            ms.float32: (np.float32, 'uniform', -9, 9),
            ms.float64: (np.float64, 'uniform', -9, 9),
            ms.complex64: (np.complex64, 'uniform', -9, 9),
            ms.complex128: (np.complex128, 'uniform', -9, 9),
            ms.bfloat16: (np.float32, 'uniform', -9, 9),
        }

        np_dtype, default_random_method, default_low, default_high = dtype_to_np_dtype_dict[dtype]
        random_method = default_random_method if random_method is None else random_method
        low = default_low if low is None else low
        high = default_high if high is None else high
        ndarray = _generate_ndarray_by_random_method(random_method, shape, np_dtype,
                                                  low, high)
        if exclude_value and dtype != ms.bool_ and 0 not in shape:
            mask = np.isin(ms.Tensor(ndarray, dtype = dtype), exclude_value)
            time = 100
            while mask.any():
                assert time > 0, "generate_ndarry_random loop count exceeds limits 100"
                ndarray[mask] = _generate_ndarray_by_random_method(random_method, mask.sum()
                                                                   , np_dtype, low, high)
                mask = np.isin(ms.Tensor(ndarray, dtype=dtype), exclude_value)
                time = time - 1
        return ndarray

    if random_seed is not None:
        np.random.seed(random_seed)

    if dtype == ms.complex64 or dtype == ms.complex128:
        real = _generate_ndarray(shape, dtype, low, high, random_method)
        imag = _generate_ndarray(shape, dtype, low, high, random_method)
        result = ms.Tensor((real + 1j * imag), dtype=dtype)
    else:
        result = ms.Tensor(_generate_ndarray(shape, dtype, low, high, random_method,
                                             exclude_value = exclude_value), dtype=dtype)

    if device is not None and device.lower() in ['ascend', 'cpu']:
        device_str = 'Ascend' if device.lower() == 'ascend' else 'CPU'
        result = result.move_to(device_str)

    if discontiguous:
        if device is not None and device.lower() != 'gpu':
            result = _tensor_to_discontiguous(result)

    return result


def make_tensor_with_np_array(
        np_array: np.ndarray,
        dtype=None,
        *,
        device: Optional[str] = None,
        discontiguous: Optional[bool] = False,
):
    """Wrap a numpy array into a MindSpore Tensor with optional device/memory tweaks."""
    result = ms.Tensor(np_array, dtype=dtype)
    if device is not None and device.lower() in ['ascend', 'cpu']:
        device_str = 'Ascend' if device.lower() == 'ascend' else 'CPU'
        result = result.move_to(device_str)
    if discontiguous:
        if device is not None and device.lower() != 'gpu':
            result = _tensor_to_discontiguous(result)

    return result


def wrap_sample_inputs(input_func, sample_names, *, skip=False,
                       loss: Optional[Union[float, Dict[str, float]]] = None):
    """
    Wrap a sample input generator to selectively tweak per-sample behavior.

    Args:
        input_func (function): sample input generator function.
        sample_names (str or list): keyword string or keyword list, used to match OpSampleInput.sample_name.
        skip (bool | List[str] | Dict[str, bool]):
            - bool: True means skip matched samples on all devices; False means don't skip (default).
            - List[str]: device keys to skip, subset of {"ascend", "ascend910b", "cpu", "gpu"}.
            - Dict[str, bool]: per-device skip map, keys in {"ascend", "ascend910b", "cpu", "gpu"}.
        loss (float | Dict[str, float], optional):
            - float: set matched samples' tolerance (rtol/atol) for all devices.
            - Dict[str, float]: per-device tolerance map, keys in {"ascend", "ascend910b", "cpu", "gpu"}.
    Returns:
        function: wrapped generator function.
    """
    if isinstance(sample_names, str):
        sample_names = [sample_names]

    def _get_device_key(device_arg: Optional[str]) -> str:
        # Prefer explicit device passed by caller; fall back to global context.
        device = device_arg or ms.context.get_context('device_target')
        device = str(device).lower()
        if device == 'ascend':
            ascend_name = MSContext.get_instance().get_ascend_soc_version()
            return 'ascend910b' if str(ascend_name).lower() == 'ascend910b' else 'ascend'
        if device == 'gpu':
            return 'gpu'
        if device == 'cpu':
            return 'cpu'
        # Unknown device target; keep original string for debugging.
        return device

    def _should_skip(device_key: str) -> bool:
        if isinstance(skip, bool):
            return skip
        if isinstance(skip, dict):
            return bool(skip.get(device_key, False))
        if isinstance(skip, List):
            return device_key in skip
        raise ValueError(f"Invalid skip type: {type(skip)}")

    def _get_loss(device_key: str) -> Optional[float]:
        if loss is None:
            return None
        if isinstance(loss, dict):
            return loss.get(device_key, None)
        return loss

    def wrapped_func(op_info, dtype=None, device=None, **kwargs):
        device_key = _get_device_key(device)
        for sample_input in input_func(op_info, dtype, device, **kwargs):
            is_match = any(name in sample_input.sample_name for name in sample_names)
            if not is_match:
                yield sample_input
                continue
            if _should_skip(device_key):
                continue

            sample_loss = _get_loss(device_key)
            if sample_loss is not None:
                sample_input.loss = sample_loss
            yield sample_input

    return wrapped_func


def is_op_input_dynamic(op_input):
    """Check if op input is dynamic."""
    DYNAMIC_RANK_DIM = -2
    DYNAMIC_SHAPE_DIM = -1
    def is_tensor_dynamic(tensor):
        return DYNAMIC_RANK_DIM in tensor.shape or DYNAMIC_SHAPE_DIM in tensor.shape

    if isinstance(op_input, ms.Tensor):
        return is_tensor_dynamic(op_input)
    if isinstance(op_input, (list, tuple)):
        result = False
        for item in op_input:
            if isinstance(item, ms.Tensor):
                result = result or is_tensor_dynamic(item)
            elif isinstance(item, (list, tuple, dict)):
                result = result or is_op_input_dynamic(item)
            # skip non-tensor scalars (e.g., int/float/None)
            if result:
                break
        return result
    if isinstance(op_input, dict):
        result = False
        for item in op_input.values():
            if isinstance(item, ms.Tensor):
                result = result or is_tensor_dynamic(item)
            elif isinstance(item, (list, tuple, dict)):
                result = result or is_op_input_dynamic(item)
            if result:
                break
        return result
    return False

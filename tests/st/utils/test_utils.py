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

from functools import wraps
import inspect
import sys
from typing import Sequence, List
from enum import Enum
import torch
import numpy as np


import mindspore as ms
from mindspore import jit, nn, Tensor

if sys.version_info >= (3, 9):
    list_annotation = list
else:
    list_annotation = List

ms.set_context(jit_syntax_level=ms.STRICT)


class Net(nn.Cell):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def construct(self, *inputs, **kwargs):
        return self.func(*inputs, **kwargs)


def run_with_cell(fn):
    if fn is None:
        raise ValueError("fn cannot be none!")

    @wraps(fn)
    def wrapper(*args, **kwargs):
        cell_obj = Net(fn)
        return cell_obj(*args, **kwargs)

    return wrapper


def run_with_mode(fn):
    if fn is None:
        raise ValueError("fn cannot be none!")

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if 'mode' not in kwargs:
            raise ValueError("mode not provided.")
        mode = kwargs['mode'].lower()
        if mode not in ['pynative', 'graph', 'kbk']:
            raise ValueError(
                "Invalid mode. Available option: ['pynative', 'graph', 'kbk'].")

        del kwargs['mode']
        if mode == "graph":
            return (jit(fn, backend="GE"))(*args, **kwargs)
        if mode == "kbk":
            return (jit(fn, backend="ms_backend", jit_level="O0"))(*args, **kwargs)
        return fn(*args, **kwargs)

    setattr(wrapper, "__wrapped_with_mode__", True)
    return wrapper


def run_with_cell_ext(jit_config=None):
    def cell_wrap_fn(fn):
        if fn is None:
            raise ValueError("fn cannot be none!")

        @wraps(fn)
        def wrapper(*args, **kwargs):
            cell_obj = Net(fn)
            if jit_config:
                cell_obj.set_jit_config(jit_config)
            return cell_obj(*args, **kwargs)

        return wrapper

    return cell_wrap_fn


def to_cell_obj(fn):
    cell_obj = Net(fn)
    return cell_obj


def compare(output, expect):
    '''
    :param output: Tensor, including tuple/list of tensor
    :param expect: Numpy array, including tuple/list of Numpy array
    :return:
    '''
    if isinstance(output, (tuple, list)):
        for o_ele, e_ele in zip(output, expect):
            compare(o_ele, e_ele)
    else:
        if expect.dtype == np.float32:
            rtol, atol = 1e-4, 1e-4
        else:
            rtol, atol = 1e-3, 1e-3
        if not np.allclose(output.asnumpy(), expect, rtol, atol, equal_nan=True):
            raise ValueError(f"compare failed \n output: {output.asnumpy()}\n expect: {expect}")


def generate_random_input(shape: Sequence[int], dtype: type = None) -> np.ndarray:
    array = np.random.randn(*shape)
    if dtype:
        array = array.astype(dtype)
    return array


def generate_random_tensor(shape: Sequence[int], dtype: ms.dtype) -> ms.Tensor:
    # Q: Why use `numpy.random.randn` to generate a random `numpy.ndarray` and then convert it into a
    #    `mindspore.Tensor` instead of directly using `mindspore.ops.StandardNormal` to generate a random
    #    `mindspore.Tensor`?
    # A: Because `mindspore.ops.StandardNormal` does not support the random seed reproduction function on the Ascend
    #    backend, which is not conducive to reproduct results. Reference
    #    https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.StandardNormal.html .
    return ms.Tensor(generate_random_input(shape)).type(dtype)


def get_inputs_np(shapes, dtypes):
    np.random.seed(10)
    inputs_np = []
    for shape, dtype in zip(shapes, dtypes):
        inputs_np.append(generate_random_input(shape, dtype))
    return inputs_np


def get_inputs_tensor(inputs_np):
    inputs = []
    for input_np in inputs_np:
        inputs.append(Tensor(input_np))
    return inputs


def convert_ms_tensor_to_numpy_array(tensor: ms.Tensor) -> np.ndarray:
    if tensor.dtype == ms.bfloat16:
        tensor = tensor.astype(ms.float32)
    return tensor.asnumpy()


def convert_ms_tensors_to_numpy_arrays(tensors: Sequence[ms.Tensor]) -> list_annotation[np.ndarray]:
    return [convert_ms_tensor_to_numpy_array(tensor) for tensor in tensors]


def need_run_graph_op_mode(func, args, kwargs):
    if ms.get_context('device_target') != 'Ascend':
        return False

    # get description of function params expected
    sig = inspect.signature(func)
    sig_args = [param.name for param in sig.parameters.values()]

    mode = None
    if isinstance(kwargs, dict):
        for key in ['mode', 'context_mode']:
            if key in sig_args and key in kwargs:
                mode = kwargs[key]
                break

    return mode == ms.GRAPH_MODE


def run_test_with_On(test_func):

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # call original test function
        test_func(*args, **kwargs)

        if not need_run_graph_op_mode(test_func, args, kwargs):
            return

        org_jit_level = ms.get_context('jit_level')
        try:
            # run graph in kernel by kernel mode
            ms.set_context(jit_level='O0')
            test_func(*args, **kwargs)
        finally:
            ms.set_context(jit_level=org_jit_level)

    return wrapper


MIN_ERR = 1e-7


def get_eb(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual.to(torch.float32) - golden
    error_balance = torch.mean(actual_error / golden_nmax)
    return error_balance


def get_mare(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    mare = torch.max(abs_error.flatten())
    return mare


def get_mere(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    actual = actual.to(torch.float32)
    abs_error = torch.abs(actual - golden) / (torch.abs(golden) + MIN_ERR)
    mere = torch.mean(abs_error)
    return mere


def get_rmse(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    sqr_err = torch.pow((actual.to(torch.float32) - golden), 2)
    rmse = torch.sqrt(torch.mean(sqr_err))
    return rmse


def get_eb_threshold(dtype: torch.dtype):
    eb_threshold = 0.0
    if dtype == torch.bfloat16:
        eb_threshold = 2 ** (-7)
    if dtype == torch.float16:
        eb_threshold = 2 ** (-10)
    elif dtype == torch.float32:
        eb_threshold = 2 ** (-14)
    return eb_threshold


class OpTypes(Enum):
    """Operation categories used to select numeric thresholds.

    Values group operators by numerical behavior to apply suitable
    comparison thresholds. These categories reflect typical error
    characteristics for different kinds of computations.

    Members:
        NA: Not applicable or unspecified.
        MOVE: Data movement only (copy, reshape, etc.).
        RAND: Random number generation.
        CAST: Data type casting.
        COMPUTE_INTEGER: Integer-domain computations.
        COMPUTE_QUANT: Quantization related computations.
        COMPUTE_FLOAT: Floating-point computations (standard precision).
        COMPUTE_FLOAT_HIGH_PRECISION: Floating-point computations requiring
            tighter tolerances.
        VECTOR_FLOAT: Vectorized floating-point ops.
        CV_FUSION: Computer-vision style fused kernels.
    """
    NA = 0
    MOVE = 1
    RAND = 2
    CAST = 3
    COMPUTE_INTEGER = 4
    COMPUTE_QUANT = 5
    COMPUTE_FLOAT = 6
    COMPUTE_FLOAT_HIGH_PRECISION = 7
    VECTOR_FLOAT = 8
    CV_FUSION = 9


def get_err_threshold(op_type: OpTypes, dtype: torch.dtype):
    err_threshold = 0.0
    if op_type in [OpTypes.COMPUTE_QUANT, OpTypes.COMPUTE_FLOAT]:
        if dtype == torch.bfloat16:
            err_threshold = 2 ** (-7)
        if dtype == torch.float16:
            err_threshold = 2 ** (-8)
        elif dtype == torch.float32:
            err_threshold = 2 ** (-11)

    if op_type in [OpTypes.CV_FUSION]:
        if dtype == torch.bfloat16:
            err_threshold = 2 ** (-8)
        if dtype == torch.float16:
            err_threshold = 2 ** (-11)
        elif dtype == torch.float32:
            err_threshold = 2 ** (-14)
    return err_threshold


def double_golden_compare(golden: torch.Tensor, gpu: torch.Tensor, actual: ms.Tensor,
                          op_type: OpTypes = OpTypes.CV_FUSION):
    """Compare ``actual`` against two baselines using CV-style metrics.

    Args:
        golden: Baseline tensor in a higher precision than ``actual``
            (e.g., float32 if actual is float16/bfloat16). Same shape as
            ``gpu`` and ``actual``.
        gpu: Baseline tensor with the same dtype as ``actual``
            (e.g., float16/bfloat16). Same shape as ``golden`` and ``actual``.
        actual: Candidate output tensor to validate. Same shape as
            ``golden``/``gpu``.
        op_type: Operation category used to select numeric thresholds. See
            ``OpTypes`` for available categories. Default: ``OpTypes.CV_FUSION``.

    Behavior:
        - Check that all shapes are equal. If not, print shapes/dtypes and
          return False.
        - Compute MARE/MERE/RMSE for (actual,golden) and (gpu,golden), then
          ratios: rate = metric(actual,golden)/max(metric(gpu,golden), eps).
        - Compute EB(actual,gpu).
        - Pass if all hold: MARE_rate < 10, MERE_rate < 2, RMSE_rate < 2,
          EB < EB_threshold(dtype(actual)).

    Returns:
        bool: True if all conditions pass, otherwise False. On failure, prints
        detailed statistics, thresholds and failed conditions to ease debug.

    Notes:
        - Thresholds are chosen by ``op_type`` and ``actual.dtype``.
        - All metric computations are internally performed in float32 for
          numerical stability.
    """
    if actual.dtype == ms.bfloat16:
        actual = torch.from_numpy(actual.to(ms.float32).asnumpy()).to(torch.bfloat16)
    else:
        actual = torch.from_numpy(actual.asnumpy())

    eb_threshold = get_eb_threshold(actual.dtype)
    err_threshold = get_err_threshold(op_type, actual.dtype)

    # Shape check first
    if golden.shape != gpu.shape or golden.shape != actual.shape:
        def _shape_msg(x):
            return "shape={}, dtype={}".format(tuple(x.shape), str(x.dtype))

        print("[compare_cv] Shape mismatch detected.")
        print("- golden:", _shape_msg(golden))
        print("- gpu   :", _shape_msg(gpu))
        print("- actual:", _shape_msg(actual))
        return False

    # All empty tensors: treat as pass
    if golden.numel() == 0 and gpu.numel() == 0 and actual.numel() == 0:
        return True

    mare_ms = get_mare(golden, actual)
    mare_gpu = get_mare(golden, gpu)

    mere_ms = get_mere(golden, actual)
    mere_gpu = get_mere(golden, gpu)

    rmse_ms = get_rmse(golden, actual)
    rmse_gpu = get_rmse(golden, gpu)

    mare_rate = mare_ms / max(mare_gpu, err_threshold)
    mere_rate = mere_ms / max(mere_gpu, err_threshold)
    rmse_rate = rmse_ms / max(rmse_gpu, err_threshold)

    eb = get_eb(gpu, actual)

    mare_ok = bool((mare_rate < 10).item())
    mere_ok = bool((mere_rate < 2).item())
    rmse_ok = bool((rmse_rate < 2).item())
    eb_ok = bool((eb < eb_threshold).item())

    result = mare_ok and mere_ok and rmse_ok and eb_ok

    if not result:
        # Summarize stats for quick diagnosis.
        def _stat(x):
            x32 = x.to(torch.float32)
            return {
                'dtype': str(x.dtype),
                'shape': tuple(x.shape),
                'min': float(torch.min(x32).item()) if x32.numel() > 0 else 0.0,
                'max': float(torch.max(x32).item()) if x32.numel() > 0 else 0.0,
                'mean': float(torch.mean(x32).item()) if x32.numel() > 0 else 0.0,
                'std': float(torch.std(x32).item()) if x32.numel() > 1 else 0.0,
            }

        abs_err = torch.abs(actual.to(torch.float32) - golden.to(torch.float32))
        max_abs_err = float(torch.max(abs_err).item()) if abs_err.numel() > 0 else 0.0
        mean_abs_err = float(torch.mean(abs_err).item()) if abs_err.numel() > 0 else 0.0

        print("[compare_cv] Accuracy check failed.")
        print("- golden stats:", _stat(golden))
        print("- gpu    stats:", _stat(gpu))
        print("- actual stats:", _stat(actual))

        print("- MARE(actual,golden)=", float(mare_ms.item()))
        print("- MARE(gpu,golden)   =", float(mare_gpu.item()))
        print("- MARE rate          =", float(mare_rate.item()))

        print("- MERE(actual,golden)=", float(mere_ms.item()))
        print("- MERE(gpu,golden)   =", float(mere_gpu.item()))
        print("- MERE rate          =", float(mere_rate.item()))

        print("- RMSE(actual,golden)=", float(rmse_ms.item()))
        print("- RMSE(gpu,golden)   =", float(rmse_gpu.item()))
        print("- RMSE rate          =", float(rmse_rate.item()))

        print("- EB(actual,gpu) =", float(eb.item()))

        print("- Max abs error      =", max_abs_err)
        print("- Mean abs error     =", mean_abs_err)

        print(f"- Thresholds: MARE rate<10, MERE rate<2, RMSE rate<2, EB<{eb_threshold}")

        fail_reasons = []
        if not mare_ok:
            fail_reasons.append(
                "MARE rate failed: {:.6g} >= 10 (MARE(actual,golden)={:.6g}, "
                "MARE(gpu,golden)={:.6g})".format(
                    float(mare_rate.item()),
                    float(mare_ms.item()),
                    float(mare_gpu.item()),
                )
            )
        if not mere_ok:
            fail_reasons.append(
                "MERE rate failed: {:.6g} >= 2 (MERE(actual,golden)={:.6g}, "
                "MERE(gpu,golden)={:.6g})".format(
                    float(mere_rate.item()),
                    float(mere_ms.item()),
                    float(mere_gpu.item()),
                )
            )
        if not rmse_ok:
            fail_reasons.append(
                "RMSE rate failed: {:.6g} >= 2 (RMSE(actual,golden)={:.6g}, "
                "RMSE(gpu,golden)={:.6g})".format(
                    float(rmse_rate.item()),
                    float(rmse_ms.item()),
                    float(rmse_gpu.item()),
                )
            )
        if not eb_ok:
            fail_reasons.append(
                "EB failed: {:.6g} >= {:.6g}".format(float(eb.item()), eb_threshold)
            )

        if fail_reasons:
            print("- Failed conditions:")
            for r in fail_reasons:
                print("  *", r)

    return result


def ref_compare(golden: torch.Tensor, actual: torch.Tensor, threshold: float):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    result = (abs_error <= threshold * golden_nmax).all()
    # Ensure Python bool return for safe logical ops
    return bool(result.item()) if torch.is_tensor(result) else bool(result)


def single_golden_compare(golden: torch.Tensor, actual: ms.Tensor, ksize,
                          op_type: OpTypes = OpTypes.CV_FUSION):
    """Compare with one ``golden`` using EB and elementwise threshold.

    Args:
        golden: Golden baseline tensor. It can be in the same precision as
            ``actual`` (e.g., both float32) or in a higher precision
            (e.g., float32 when ``actual`` is float16/bfloat16/hf32).
        actual: Candidate output tensor to validate.
        ksize: Problem/kernel size that controls compare threshold scaling.
            When ``ksize >= 2048`` the elementwise threshold is multiplied by
            10; otherwise the base threshold is used.
        op_type: Operation category used to select numeric thresholds. See
            ``OpTypes`` for available categories. Default: ``OpTypes.CV_FUSION``.

    Behavior:
        - Check shapes equality; on mismatch print shapes/dtypes and return
          False.
        - Compute EB(actual,golden) and an elementwise compare:
          |actual - golden| <= threshold * max(|golden|, 1).
        - ``threshold`` is selected by ``op_type`` and dtype.

    Returns:
        bool: True if EB and elementwise checks both pass; else False. Prints
        detailed statistics, thresholds and failure counts on mismatch.
    """
    if actual.dtype == ms.bfloat16:
        actual = torch.from_numpy(actual.to(ms.float32).asnumpy()).to(torch.bfloat16)
    else:
        actual = torch.from_numpy(actual.asnumpy())

    eb_threshold = get_eb_threshold(actual.dtype)
    err_threshold = get_err_threshold(op_type, actual.dtype)

    # Shape check first
    if golden.shape != actual.shape:
        def _shape_msg(x):
            return "shape={}, dtype={}".format(tuple(x.shape), str(x.dtype))

        print("[single_golden_compare] Shape mismatch detected.")
        print("- golden:", _shape_msg(golden))
        print("- actual:", _shape_msg(actual))
        return False

    # Both empty tensors: treat as pass
    if golden.numel() == 0 and actual.numel() == 0:
        return True

    threshold = err_threshold if ksize < 2048 else err_threshold * 10

    eb = get_eb(golden, actual)
    cmp = ref_compare(golden, actual, threshold)

    eb_ok = bool((eb < eb_threshold).item())
    cmp_ok = bool(cmp)
    result = eb_ok and cmp_ok

    if not result:
        def _stat(x):
            x32 = x.to(torch.float32)
            return {
                'dtype': str(x.dtype),
                'shape': tuple(x.shape),
                'min': float(torch.min(x32).item()) if x32.numel() > 0 else 0.0,
                'max': float(torch.max(x32).item()) if x32.numel() > 0 else 0.0,
                'mean': float(torch.mean(x32).item()) if x32.numel() > 0 else 0.0,
                'std': float(torch.std(x32).item()) if x32.numel() > 1 else 0.0,
            }

        golden32 = golden.to(torch.float32)
        actual32 = actual.to(torch.float32)
        golden_nmax = torch.clamp(torch.abs(golden32), min=1)
        abs_error = torch.abs(actual32 - golden32)
        rel_error = abs_error / golden_nmax

        max_abs_err = float(torch.max(abs_error).item()) if abs_error.numel() > 0 else 0.0
        mean_abs_err = float(torch.mean(abs_error).item()) if abs_error.numel() > 0 else 0.0
        max_rel_err = float(torch.max(rel_error).item()) if rel_error.numel() > 0 else 0.0

        fail_mask = abs_error > (threshold * golden_nmax)
        num_fail = int(torch.sum(fail_mask).item()) if fail_mask.numel() > 0 else 0
        num_total = int(abs_error.numel())

        print("[single_golden_compare] Accuracy check failed.")
        print("- golden stats:", _stat(golden))
        print("- actual stats:", _stat(actual))
        print("- EB(actual,golden) =", float(eb.item()))
        print("- EB threshold      =", eb_threshold)
        print("- compare threshold =", threshold)
        print("- Max abs error     =", max_abs_err)
        print("- Mean abs error    =", mean_abs_err)
        print("- Max rel error     =", max_rel_err)
        print("- Fail count/total  =", num_fail, "/", num_total)

        fail_reasons = []
        if not eb_ok:
            fail_reasons.append(
                "EB failed: {:.6g} >= {:.6g}".format(
                    float(eb.item()), eb_threshold
                )
            )
        if not cmp_ok:
            fail_reasons.append(
                "Elementwise compare failed: {} of {} elements exceed threshold".format(
                    num_fail, num_total
                )
            )
        if fail_reasons:
            print("- Failed conditions:")
            for r in fail_reasons:
                print("  *", r)

    return result

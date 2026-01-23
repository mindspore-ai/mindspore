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
""" test_nsa_compress.py """
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, ops
from mindspore import mint
from mindspore import context
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import single_golden_compare
from tests.st.ops.test_tools.test_op import TEST_OP
import torch


def nsa_compress_forward_func(x, w, block, stride, actual_seq_len):
    """Forward function for nsa_compress."""
    return ops.nsa_compress(x, w, block, stride, actual_seq_len=actual_seq_len)


def nsa_compress_backward_func(x, w, block, stride, actual_seq_len):
    """Backward function for nsa_compress."""
    return ms.grad(nsa_compress_forward_func, (0, 1))(
        x, w, block, stride, actual_seq_len
    )


@ms.jit
def nsa_compress_forward_func_jit(x, w, block, stride, actual_seq_len):
    return ops.nsa_compress(x, w, block, stride, actual_seq_len=actual_seq_len)


@ms.jit
def nsa_compress_backward_func_jit(x, w, block, stride, actual_seq_len):
    return ms.grad(nsa_compress_forward_func_jit, (0, 1))(
        x, w, block, stride, actual_seq_len
    )


def _ms_forward_backward(context_mode, x_np, w_np, block, stride, actual_seq_len, dtype=ms.float16):
    """Run forward and backward in MindSpore with numpy inputs, return MindSpore Tensors (y, gx, gw)."""
    context.set_context(mode=context_mode)
    x = Tensor(x_np, dtype=dtype)
    w = Tensor(w_np, dtype=dtype)
    if context_mode == ms.GRAPH_MODE:
        y = nsa_compress_forward_func_jit(x, w, block, stride, actual_seq_len)
        if int(y.shape[0]) == 0:
            gx = ops.zeros_like(x)
            gw = ops.zeros_like(w)
            return y, gx, gw
        gx, gw = nsa_compress_backward_func_jit(x, w, block, stride, actual_seq_len)
    else:
        y = nsa_compress_forward_func(x, w, block, stride, actual_seq_len)
        if int(y.shape[0]) == 0:
            gx = ops.zeros_like(x)
            gw = ops.zeros_like(w)
            return y, gx, gw
        gx, gw = nsa_compress_backward_func(x, w, block, stride, actual_seq_len)
    return y, gx, gw


def _torch_cpu_forward_backward(x_np, w_np, block, stride, actual_seq_len, dtype=torch.float32):
    x = torch.from_numpy(x_np).to(dtype)
    w = torch.from_numpy(w_np).to(dtype)
    x.requires_grad_(True)
    w.requires_grad_(True)
    tokens = []
    pre = 0
    w_expand = w.unsqueeze(-1).expand(-1, -1, x.shape[2])
    for i, xend in enumerate(actual_seq_len):
        cur = int(xend) - int(pre)
        pre = int(xend)
        if cur < block:
            continue
        for start in range(0, cur, stride):
            if start + block > cur:
                break
            start_global = start + (int(actual_seq_len[i - 1]) if i != 0 else 0)
            window = x[start_global:start_global + block]
            tokens.append(torch.sum(window * w_expand, dim=0))

    if not tokens:
        y = torch.zeros((0, x.shape[1], x.shape[2]), dtype=dtype)
        gx = torch.zeros_like(x)
        gw = torch.zeros_like(w)
        return (y.detach().cpu(), gx.detach().cpu(), gw.detach().cpu())

    y = torch.stack(tokens, dim=0)
    y.backward(torch.ones_like(y))
    return (y.detach().cpu(), x.grad.detach().cpu(), w.grad.detach().cpu())


def _gen_inputs_for_compare(N, D, block, stride, T_max=400, num_samples=24, seed=2025, dtype=np.float16):
    rng = np.random.default_rng(seed)
    per = rng.integers(
        low=0,
        high=max(1, T_max // max(1, num_samples) + 2),
        size=(num_samples,),
        dtype=np.int64,
    )
    # Ensure at least one segment length >= block to avoid empty output
    if np.max(per) < block:
        per[-1] = block
    actual_seq = np.cumsum(per, dtype=np.int64)
    if actual_seq[-1] == 0:
        actual_seq[-1] = block
    T_total = int(actual_seq[-1])
    x_np = rng.standard_normal((T_total, N, D)).astype(dtype)
    w_np = rng.standard_normal((block, N)).astype(dtype)
    return x_np, w_np, tuple(int(v) for v in actual_seq)


def _gen_zero_token_case(N, D, block, stride, num_samples=8, seed=7,
                         dtype=np.float16):
    """Generate inputs that produce empty output tokens.

    Ensure every segment length is smaller than block so no window fits.
    """
    rng = np.random.default_rng(seed)
    per = rng.integers(low=0, high=max(1, block - 1), size=(num_samples,),
                       dtype=np.int64)
    # Ensure at least one non-zero segment but still < block
    if int(np.sum(per)) == 0:
        per[-1] = min(block - 1, 1)
    if np.max(per) >= block:
        per[:] = min(int(block - 1), 1)
    actual_seq = np.cumsum(per, dtype=np.int64)
    T_total = int(actual_seq[-1]) if actual_seq[-1] > 0 else 0
    x_np = (rng.standard_normal((T_total, N, D)).astype(dtype)
            if T_total > 0 else np.zeros((0, N, D), dtype=dtype))
    w_np = rng.standard_normal((block, N)).astype(dtype)
    return x_np, w_np, tuple(int(v) for v in actual_seq)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=20.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_cv_metric_against_torch_cpu(context_mode):
    """
    Feature: NSACompress forward/backward numerical accuracy
    Description: Compare MindSpore results with a high-precision CPU (float32) golden for y/gx/gw
    Expectation: Shapes and values exactly match the golden in Pynative and Graph O0
    """
    # generate inputs
    N, D, block, stride = 4, 128, 16, 16
    x_np, w_np, actual_seq = _gen_inputs_for_compare(
        N, D, block, stride, T_max=400, num_samples=24, seed=1011,
        dtype=np.float16,
    )

    # MindSpore outputs (float16)
    ms_y, ms_gx, ms_gw = _ms_forward_backward(
        context_mode, x_np, w_np, block, stride, actual_seq, ms.float16,
    )

    # Torch CPU baselines
    # 1) High precision baseline (float32)
    th_y_fp32, th_gx_fp32, th_gw_fp32 = _torch_cpu_forward_backward(
        x_np, w_np, block, stride, list(actual_seq), torch.float32
    )

    golden_y = th_y_fp32
    actual_y = ms_y

    golden_gx = th_gx_fp32
    actual_gx = ms_gx

    golden_gw = th_gw_fp32
    actual_gw = ms_gw

    assert single_golden_compare(golden_y, actual_y, D)
    assert single_golden_compare(golden_gx, actual_gx, D)
    assert single_golden_compare(golden_gw, actual_gw, D)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=366.00
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize(
    "case_conf",
    [
        # basic valid
        {"N": 4, "D": 128, "block": 16, "stride": 16, "T_max": 512, "num_samples": 24,
         "dtype": np.float16, "seed": 1001},
        # larger stride smaller block windows
        {"N": 8, "D": 128, "block": 32, "stride": 16, "T_max": 768, "num_samples": 30,
         "dtype": np.float16, "seed": 2027},
        # larger block
        {"N": 4, "D": 128, "block": 64, "stride": 16, "T_max": 800, "num_samples": 28,
         "dtype": np.float16, "seed": 2},
        # max-ish dims
        {"N": 128, "D": 256, "block": 128, "stride": 16, "T_max": 2048, "num_samples": 64,
         "dtype": np.float16, "seed": 3},
        # bf16 path (inputs via fp32 then cast internally)
        {"N": 4, "D": 128, "block": 16, "stride": 16, "T_max": 400, "num_samples": 16,
         "dtype": np.float32, "seed": 22, "ms_dtype": ms.bfloat16},
    ],
)
def test_nsa_compress_functional_general(context_mode, case_conf):
    """
    Feature: Functional coverage across representative specs and data types
    Description: Run multiple configurations including bf16 and compare to float32 CPU golden
    Expectation: y/gx/gw exactly match golden and shapes align in both execution modes
    """
    N = case_conf["N"]
    D = case_conf["D"]
    block = case_conf["block"]
    stride = case_conf["stride"]
    T_max = case_conf.get("T_max", 400)
    num_samples = case_conf.get("num_samples", 24)
    dtype_np = case_conf.get("dtype", np.float16)
    seed = case_conf.get("seed", 0)
    ms_dtype = case_conf.get("ms_dtype", ms.float16)

    x_np, w_np, actual_seq = _gen_inputs_for_compare(
        N, D, block, stride, T_max=T_max, num_samples=num_samples,
        seed=seed, dtype=dtype_np,
    )

    if ms_dtype == ms.bfloat16:
        x_np = torch.from_numpy(x_np).to(torch.bfloat16).to(torch.float32).numpy()
        w_np = torch.from_numpy(w_np).to(torch.bfloat16).to(torch.float32).numpy()

    # MindSpore
    ms_y, ms_gx, ms_gw = _ms_forward_backward(
        context_mode, x_np, w_np, block, stride, actual_seq, ms_dtype,
    )

    # Torch CPU baselines: golden(high fp32) and gpu(same as ms output dtype)
    th_y_fp32, th_gx_fp32, th_gw_fp32 = _torch_cpu_forward_backward(
        x_np, w_np, block, stride, list(actual_seq), torch.float32
    )

    # Compare by single_golden_compare (golden could be higher precision)
    assert single_golden_compare(th_y_fp32, ms_y, D)
    assert single_golden_compare(th_gx_fp32, ms_gx, D)
    assert single_golden_compare(th_gw_fp32, ms_gw, D)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=20.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_zero_token_outputs(context_mode):
    """
    Feature: Empty output handling (no valid compression windows)
    Description: Build seq so every segment < block, producing T' == 0; skip backward when empty
    Expectation: y has T' == 0 and gradients are zeros; golden comparison succeeds
    """
    # Intentionally construct zero-token case
    N, D, block, stride = 4, 128, 32, 16
    x_np, w_np, actual_seq = _gen_zero_token_case(N, D, block, stride, num_samples=8, seed=9)

    ms_y, ms_gx, ms_gw = _ms_forward_backward(
        context_mode, x_np, w_np, block, stride, actual_seq, ms.float16,
    )

    # golden from torch fp32
    th_y_fp32, th_gx_fp32, th_gw_fp32 = _torch_cpu_forward_backward(
        x_np, w_np, block, stride, list(actual_seq), torch.float32
    )

    # empty outputs still should pass tolerance checks
    assert single_golden_compare(th_y_fp32, ms_y, D)
    assert single_golden_compare(th_gx_fp32, ms_gx, D)
    assert single_golden_compare(th_gw_fp32, ms_gw, D)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=20.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_stride_eq_block_and_min_dims(context_mode):
    """
    Feature: stride == block with minimal dimensions
    Description: Validate correctness for N=1, D=16 and step-aligned tiling
    Expectation: Exact match with CPU golden across forward and backward
    """
    # stride == block, and minimal dims N=1, D=16
    N, D, block, stride = 1, 16, 32, 32
    x_np, w_np, actual_seq = _gen_inputs_for_compare(
        N, D, block, stride, T_max=256, num_samples=20, seed=404, dtype=np.float16
    )

    ms_y, ms_gx, ms_gw = _ms_forward_backward(
        context_mode, x_np, w_np, block, stride, actual_seq, ms.float16,
    )
    th_y_fp32, th_gx_fp32, th_gw_fp32 = _torch_cpu_forward_backward(
        x_np, w_np, block, stride, list(actual_seq), torch.float32
    )

    assert single_golden_compare(th_y_fp32, ms_y, D)
    assert single_golden_compare(th_gx_fp32, ms_gx, D)
    assert single_golden_compare(th_gw_fp32, ms_gw, D)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=24.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_bfloat16_large_dims(context_mode):
    """
    Feature: bfloat16 execution on larger shapes
    Description: Compute in bf16 (inputs prepared via fp32) and compare to float32 golden
    Expectation: Exact match after dtype alignment; shapes identical
    """
    # bf16 with larger dims
    N, D, block, stride = 8, 128, 32, 16
    x_np, w_np, actual_seq = _gen_inputs_for_compare(
        N, D, block, stride, T_max=768, num_samples=40, seed=505, dtype=np.float32
    )

    x_np = torch.from_numpy(x_np).to(torch.bfloat16).to(torch.float32).numpy()
    w_np = torch.from_numpy(w_np).to(torch.bfloat16).to(torch.float32).numpy()

    ms_y, ms_gx, ms_gw = _ms_forward_backward(
        context_mode, x_np, w_np, block, stride, actual_seq, ms.bfloat16,
    )
    th_y_fp32, th_gx_fp32, th_gw_fp32 = _torch_cpu_forward_backward(
        x_np, w_np, block, stride, list(actual_seq), torch.float32
    )
    assert single_golden_compare(th_y_fp32, ms_y, D)
    assert single_golden_compare(th_gx_fp32, ms_gx, D)
    assert single_golden_compare(th_gw_fp32, ms_gw, D)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=20.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_seq_len_list_and_window_alignment(context_mode):
    """
    Feature: List-based actual_seq_len and window-edge alignment
    Description: Mix exact-block and near-block segments to exercise boundary behavior
    Expectation: Exact match with CPU golden for y/gx/gw
    """
    # Use list for seq len and craft segments on window edges
    N, D, block, stride = 4, 128, 32, 16
    # build per segments to create exact 1-window and non-aligned windows
    per = [32, 48, 63, 16, 17, 31]  # mix of exact block, multi, and <block
    actual_seq = np.cumsum(np.array(per, dtype=np.int64))
    T_total = int(actual_seq[-1])
    rng = np.random.default_rng(606)
    x_np = rng.standard_normal((T_total, N, D)).astype(np.float16)
    w_np = rng.standard_normal((block, N)).astype(np.float16)

    ms_y, ms_gx, ms_gw = _ms_forward_backward(
        context_mode, x_np, w_np, block, stride, actual_seq.tolist(), ms.float16,
    )
    th_y_fp32, th_gx_fp32, th_gw_fp32 = _torch_cpu_forward_backward(
        x_np, w_np, block, stride, list(actual_seq.tolist()), torch.float32
    )
    assert single_golden_compare(th_y_fp32, ms_y, D)
    assert single_golden_compare(th_gx_fp32, ms_gx, D)
    assert single_golden_compare(th_gw_fp32, ms_gw, D)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=0.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_exception_none_seq_len(context_mode):
    """
    Feature: Argument validation for actual_seq_len
    Description: actual_seq_len is None
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    N, D, block, stride = 4, 128, 16, 16
    rng = np.random.default_rng(100)
    T = 256
    x_np = rng.standard_normal((T, N, D)).astype(np.float16)
    w_np = rng.standard_normal((block, N)).astype(np.float16)
    x = Tensor(x_np, dtype=ms.float16)
    w = Tensor(w_np, dtype=ms.float16)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress(x, w, block, stride, actual_seq_len=None)
        y.asnumpy()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_exception_weight_n_mismatch(context_mode):
    """
    Feature: Input consistency validation (weight N must match input N)
    Description: Construct weight with mismatched N
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    N, D, block, stride = 4, 128, 16, 16
    rng = np.random.default_rng(101)
    per = rng.integers(low=block, high=block + 8, size=(8,), dtype=np.int64)
    actual_seq = tuple(np.cumsum(per, dtype=np.int64).tolist())
    T = int(actual_seq[-1])
    x_np = rng.standard_normal((T, N, D)).astype(np.float16)
    # make weight N mismatched (N+1)
    w_np = rng.standard_normal((block, N + 1)).astype(np.float16)
    x = Tensor(x_np, dtype=ms.float16)
    w = Tensor(w_np, dtype=ms.float16)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress(x, w, block, stride, actual_seq_len=actual_seq)
        y.asnumpy()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_exception_block_stride_invalid(context_mode):
    """
    Feature: Parameter constraints for block/stride
    Description: Use values not multiples of 16 or out of valid ranges
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    N, D, block, stride = 4, 128, 18, 10  # not multiples of 16
    rng = np.random.default_rng(102)
    per = rng.integers(low=block, high=block + 8, size=(6,), dtype=np.int64)
    actual_seq = tuple(np.cumsum(per, dtype=np.int64).tolist())
    T = int(actual_seq[-1])
    x_np = rng.standard_normal((T, N, D)).astype(np.float16)
    w_np = rng.standard_normal((block, N)).astype(np.float16)
    x = Tensor(x_np, dtype=ms.float16)
    w = Tensor(w_np, dtype=ms.float16)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress(x, w, block, stride, actual_seq_len=actual_seq)
        y.asnumpy()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_exception_stride_gt_block(context_mode):
    """
    Feature: Parameter constraint stride <= block
    Description: Set stride greater than block
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    N, D, block, stride = 4, 128, 16, 32
    rng = np.random.default_rng(103)
    per = rng.integers(low=block, high=block + 8, size=(6,), dtype=np.int64)
    actual_seq = tuple(np.cumsum(per, dtype=np.int64).tolist())
    T = int(actual_seq[-1])
    x_np = rng.standard_normal((T, N, D)).astype(np.float16)
    w_np = rng.standard_normal((block, N)).astype(np.float16)
    x = Tensor(x_np, dtype=ms.float16)
    w = Tensor(w_np, dtype=ms.float16)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress(x, w, block, stride, actual_seq_len=actual_seq)
        y.asnumpy()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=1.00
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_exception_d_not_multiple_16(context_mode):
    """
    Feature: Input channel alignment D % 16 == 0
    Description: Use D not divisible by 16
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    N, D, block, stride = 4, 18, 16, 16  # D not multiple of 16
    rng = np.random.default_rng(104)
    per = rng.integers(low=block, high=block + 8, size=(6,), dtype=np.int64)
    actual_seq = tuple(np.cumsum(per, dtype=np.int64).tolist())
    T = int(actual_seq[-1])
    x_np = rng.standard_normal((T, N, D)).astype(np.float16)
    w_np = rng.standard_normal((block, N)).astype(np.float16)
    x = Tensor(x_np, dtype=ms.float16)
    w = Tensor(w_np, dtype=ms.float16)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress(x, w, block, stride, actual_seq_len=actual_seq)
        y.asnumpy()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=0.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_exception_dtype_mismatch(context_mode):
    """
    Feature: Dtype validation and consistency
    Description: input and weight have different dtypes
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    N, D, block, stride = 4, 128, 16, 16
    rng = np.random.default_rng(105)
    per = rng.integers(low=block, high=block + 8, size=(6,), dtype=np.int64)
    actual_seq = tuple(np.cumsum(per, dtype=np.int64).tolist())
    T = int(actual_seq[-1])
    x_np = rng.standard_normal((T, N, D)).astype(np.float16)
    w_np = rng.standard_normal((block, N)).astype(np.float32)  # mismatch
    x = Tensor(x_np, dtype=ms.float16)
    w = Tensor(w_np, dtype=ms.float32)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress(x, w, block, stride, actual_seq)
        y.asnumpy()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=1.00
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_exception_seq_len_type(context_mode):
    """
    Feature: actual_seq_len type validation
    Description: Provide a sequence with float values instead of integers
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    N, D, block, stride = 4, 128, 16, 16
    rng = np.random.default_rng(106)
    per = rng.integers(low=block, high=block + 8, size=(6,), dtype=np.int64)
    actual_seq_bad = tuple((np.cumsum(per, dtype=np.int64).astype(np.float32)).tolist())
    T = int(actual_seq_bad[-1])
    x_np = rng.standard_normal((T, N, D)).astype(np.float16)
    w_np = rng.standard_normal((block, N)).astype(np.float16)
    x = Tensor(x_np, dtype=ms.float16)
    w = Tensor(w_np, dtype=ms.float16)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress(x, w, block, stride, actual_seq_len=actual_seq_bad)
        y.asnumpy()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=1.00
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_exception_seq_len_not_match_T(context_mode):
    """
    Feature: actual_seq_len last element must equal T
    Description: Make the last element not equal to input length T
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    N, D, block, stride = 4, 128, 16, 16
    rng = np.random.default_rng(107)
    per = rng.integers(low=block, high=block + 8, size=(6,), dtype=np.int64)
    good_seq = np.cumsum(per, dtype=np.int64)
    bad_seq = tuple((good_seq + 1).tolist())  # last value mismatch T
    T = int(good_seq[-1])
    x_np = rng.standard_normal((T, N, D)).astype(np.float16)
    w_np = rng.standard_normal((block, N)).astype(np.float16)
    x = Tensor(x_np, dtype=ms.float16)
    w = Tensor(w_np, dtype=ms.float16)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress(x, w, block, stride, actual_seq_len=bad_seq)
        y.asnumpy()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential", mem_peak=24.00)
def test_nsa_compress_dynamic_shape_TEST_OP():
    """
    Feature: Dynamic-shape verification via TEST_OP
    Description: Run two representative input sets to exercise dynamic behavior
    Expectation: Executes successfully with correct output shapes; no crash
    """
    # Case 1
    N1, D1, block1, stride1 = 4, 128, 16, 16
    rng1 = np.random.default_rng(7001)
    per1 = rng1.integers(low=block1, high=block1 + 8, size=(12,), dtype=np.int64)
    seq1 = np.cumsum(per1, dtype=np.int64).tolist()
    T1 = int(seq1[-1])
    x1 = rng1.standard_normal((T1, N1, D1)).astype(np.float16)
    w1 = rng1.standard_normal((block1, N1)).astype(np.float16)

    # Case 2
    N2, D2, block2, stride2 = 8, 128, 32, 16
    rng2 = np.random.default_rng(7002)
    per2 = rng2.integers(low=block2, high=block2 + 16, size=(10,), dtype=np.int64)
    seq2 = np.cumsum(per2, dtype=np.int64).tolist()
    T2 = int(seq2[-1])
    x2 = rng2.standard_normal((T2, N2, D2)).astype(np.float16)
    w2 = rng2.standard_normal((block2, N2)).astype(np.float16)

    TEST_OP(
        nsa_compress_forward_func,
        [[Tensor(x1, ms.float16), Tensor(w1, ms.float16), block1, stride1, seq1],
         [Tensor(x2, ms.float16), Tensor(w2, ms.float16), block2, stride2, seq2]],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=["EmptyTensor", "ScalarTensor"],
        case_config={"disable_input_check": True},
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=0.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_exception_empty_input_tensor(context_mode):
    """
    Feature: Empty input tensor validation
    Description: Input x has T == 0
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    N, D, block, stride = 4, 128, 16, 16
    # empty input T=0
    x_np = np.zeros((0, N, D), dtype=np.float16)
    w_np = np.random.default_rng(1).standard_normal((block, N)).astype(np.float16)
    x = Tensor(x_np, dtype=ms.float16)
    w = Tensor(w_np, dtype=ms.float16)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress(x, w, block, stride, (0,))
        y.asnumpy()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=1.00
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_exception_empty_weight_tensor(context_mode):
    """
    Feature: Empty weight tensor validation
    Description: Weight has shape (0, N)
    Expectation: Raises ValueError/RuntimeError/TypeError
    """
    context.set_context(mode=context_mode)
    N, D, block, stride = 4, 128, 16, 16
    rng = np.random.default_rng(10)
    per = rng.integers(low=block, high=block + 8, size=(6,), dtype=np.int64)
    seq = tuple(np.cumsum(per, dtype=np.int64).tolist())
    T = int(seq[-1])
    x_np = rng.standard_normal((T, N, D)).astype(np.float16)
    w_np = np.zeros((0, N), dtype=np.float16)
    x = Tensor(x_np, dtype=ms.float16)
    w = Tensor(w_np, dtype=ms.float16)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        y = ops.nsa_compress(x, w, block, stride, seq)
        y.asnumpy()


def _calc_tokens(seq, block, stride):
    pre = 0
    tokens = 0
    for s in seq:
        cur = int(s) - pre
        pre = int(s)
        if cur >= block:
            tokens += (cur - block + stride) // stride
    return tokens


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=18.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_inf_inputs_forward_only(context_mode):
    """
    Feature: Robustness with special values (inf)
    Description: Forward-only run with an inf value in input
    Expectation: Produces a valid output tensor with expected shape; no error
    """
    context.set_context(mode=context_mode)
    N, D, block, stride = 4, 128, 16, 16
    rng = np.random.default_rng(11)
    per = rng.integers(low=block, high=block + 8, size=(6,), dtype=np.int64)
    seq = tuple(np.cumsum(per, dtype=np.int64).tolist())
    T = int(seq[-1])
    x_np = rng.standard_normal((T, N, D)).astype(np.float16)
    x_np[0, 0, 0] = np.inf
    w_np = rng.standard_normal((block, N)).astype(np.float16)
    x = Tensor(x_np, dtype=ms.float16)
    w = Tensor(w_np, dtype=ms.float16)
    y = ops.nsa_compress(x, w, block, stride, actual_seq_len=seq)
    # shape sanity
    expect_t = _calc_tokens(seq, block, stride)
    assert tuple(y.shape) == (expect_t, N, D)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=18.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_nan_inputs_forward_only(context_mode):
    """
    Feature: Robustness with special values (nan)
    Description: Forward-only run with a nan value in input
    Expectation: Produces a valid output tensor with expected shape; no error
    """
    context.set_context(mode=context_mode)
    N, D, block, stride = 4, 128, 16, 16
    rng = np.random.default_rng(12)
    per = rng.integers(low=block, high=block + 8, size=(6,), dtype=np.int64)
    seq = tuple(np.cumsum(per, dtype=np.int64).tolist())
    T = int(seq[-1])
    x_np = rng.standard_normal((T, N, D)).astype(np.float16)
    x_np[0, 0, 1] = np.nan
    w_np = rng.standard_normal((block, N)).astype(np.float16)
    x = Tensor(x_np, dtype=ms.float16)
    w = Tensor(w_np, dtype=ms.float16)
    y = ops.nsa_compress(x, w, block, stride, actual_seq_len=seq)
    expect_t = _calc_tokens(seq, block, stride)
    assert tuple(y.shape) == (expect_t, N, D)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=22.01
)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nsa_compress_non_contiguous_inputs(context_mode):
    """
    Feature: Support for non-contiguous inputs (views via transpose)
    Description: Build non-contiguous x (T,D,N -> T,N,D) and w (N,block -> block,N) using transpose views
    Expectation: Forward/backward exactly match CPU golden; shapes identical in both modes
    """
    # base specs
    N, D, block, stride = 8, 128, 32, 16

    # build per-segment lengths to ensure at least one valid window
    rng = np.random.default_rng(2025)
    per = rng.integers(low=block, high=block + 12, size=(6,), dtype=np.int64)
    actual_seq = tuple(np.cumsum(per, dtype=np.int64).tolist())
    T = int(actual_seq[-1])

    # construct non-contiguous values using transpose views
    # build full arrays with transposed base, then transpose to target shapes to ensure non-contiguity
    x_full_np = rng.standard_normal((T, D, N)).astype(np.float16)
    w_full_np = rng.standard_normal((N, block)).astype(np.float16)
    x_np = np.transpose(x_full_np, (0, 2, 1))  # (T, N, D)
    w_np = np.transpose(w_full_np, (1, 0))     # (block, N)

    # MindSpore non-contiguous tensors via mint.transpose
    x_full_ms = Tensor(x_full_np, dtype=ms.float16)  # (T, D, N)
    w_full_ms = Tensor(w_full_np, dtype=ms.float16)  # (N, block)
    x_ms_nc = mint.transpose(x_full_ms, 1, 2)        # (T, N, D) non-contiguous view
    w_ms_nc = mint.transpose(w_full_ms, 0, 1)        # (block, N) non-contiguous view
    assert not x_ms_nc.is_contiguous()
    assert not w_ms_nc.is_contiguous()

    # MindSpore forward/backward
    context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        y = nsa_compress_forward_func_jit(x_ms_nc, w_ms_nc, block, stride, actual_seq)
        if int(y.shape[0]) == 0:
            gx = ops.zeros_like(x_ms_nc)
            gw = ops.zeros_like(w_ms_nc)
        else:
            gx, gw = nsa_compress_backward_func_jit(x_ms_nc, w_ms_nc, block, stride, actual_seq)
    else:
        y = nsa_compress_forward_func(x_ms_nc, w_ms_nc, block, stride, actual_seq)
        if int(y.shape[0]) == 0:
            gx = ops.zeros_like(x_ms_nc)
            gw = ops.zeros_like(w_ms_nc)
        else:
            gx, gw = nsa_compress_backward_func(x_ms_nc, w_ms_nc, block, stride, actual_seq)

    # CPU golden (float32)
    th_y_fp32, th_gx_fp32, th_gw_fp32 = _torch_cpu_forward_backward(
        x_np, w_np, block, stride, list(actual_seq), torch.float32
    )

    # compare
    assert single_golden_compare(th_y_fp32, y, D)
    assert single_golden_compare(th_gx_fp32, gx, D)
    assert single_golden_compare(th_gw_fp32, gw, D)

"""Tests for flash_attention_score exceptions."""
# Copyright 2024 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest
from mindspore.graph.api import _pynative_executor

import mindspore as ms
from mindspore import Tensor, context
from mindspore.ops import flash_attention_score
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils


def set_mode(mode):
    """
    Set context mode for test.
    """
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_level='O0')
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


def _gen_bsh_inputs(batch=2, seq=16, heads=4, dim=8):
    """Generate simple BSH layout inputs with float16 dtype."""
    q = np.random.uniform(-1.0, 1.0, (batch, seq, heads * dim))
    k = np.random.uniform(-1.0, 1.0, (batch, seq, heads * dim))
    v = np.random.uniform(-1.0, 1.0, (batch, seq, heads * dim))
    return Tensor(q.astype(np.float16), ms.float16), \
        Tensor(k.astype(np.float16), ms.float16), \
        Tensor(v.astype(np.float16), ms.float16)


@test_utils.run_with_cell
def _fas_call(query, key, value, head_num, real_shift=None, drop_mask=None,
              padding_mask=None, attn_mask=None, prefix=None,
              actual_seq_qlen=None, actual_seq_kvlen=None, keep_prob=1.0,
              input_layout='BSH', pre_tokens=2147483647,
              next_tokens=2147483647, scalar_value=1.0,
              inner_precise=0, sparse_mode=0):
    return flash_attention_score(query, key, value, head_num, real_shift,
                                 drop_mask, padding_mask, attn_mask, prefix,
                                 actual_seq_qlen, actual_seq_kvlen,
                                 keep_prob, scalar_value, pre_tokens,
                                 next_tokens, inner_precise, input_layout,
                                 sparse_mode)


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=0.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_keep_prob_zero_with_drop_mask_raises(mode):
    """
    Feature: flash_attention_score
    Description: keep_prob=0.0 with provided drop_mask should raise at runtime
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    q, k, v = _gen_bsh_inputs()
    # For B=2, N=4, S=16, KV=16 -> KV/8=2
    drop_mask = Tensor(np.zeros((2, 4, 16, 2), dtype=np.uint8))
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, 4, drop_mask=drop_mask, keep_prob=0.0,
                      input_layout='BSH', sparse_mode=0)
        _pynative_executor.sync()


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential',
    mem_peak=2.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_tnd_missing_actual_seq_raises(mode):
    """
    Feature: flash_attention_score
    Description: TND layout without actual_seq_qlen/actual_seq_kvlen
                  should raise at runtime
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    tdim, heads, dim = 32, 2, 8
    q = Tensor(np.random.uniform(-1.0, 1.0, (tdim, heads, dim)).astype(
        np.float16), ms.float16)
    k = Tensor(np.random.uniform(-1.0, 1.0, (tdim, heads, dim)).astype(
        np.float16), ms.float16)
    v = Tensor(np.random.uniform(-1.0, 1.0, (tdim, heads, dim)).astype(
        np.float16), ms.float16)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, heads, input_layout='TND', sparse_mode=0)
        _pynative_executor.sync()


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential',
    mem_peak=2.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_tnd_actual_seq_qlen_none_raises(mode):
    """
    Feature: flash_attention_score
    Description: TND layout with actual_seq_qlen=None but actual_seq_kvlen
                  provided should raise at runtime
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    tdim, heads, dim = 32, 2, 8
    q = Tensor(np.random.uniform(-1.0, 1.0, (tdim, heads, dim)).astype(
        np.float16), ms.float16)
    k = Tensor(np.random.uniform(-1.0, 1.0, (tdim, heads, dim)).astype(
        np.float16), ms.float16)
    v = Tensor(np.random.uniform(-1.0, 1.0, (tdim, heads, dim)).astype(
        np.float16), ms.float16)
    aseq_kv = Tensor(np.array([16, 32], dtype=np.int64), ms.int64)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, heads, actual_seq_kvlen=aseq_kv,
                      input_layout='TND', sparse_mode=0)
        _pynative_executor.sync()


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_tnd_actual_seq_kvlen_none_raises(mode):
    """
    Feature: flash_attention_score
    Description: TND layout with actual_seq_kvlen=None but actual_seq_qlen
                  provided should raise at runtime
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    tdim, heads, dim = 32, 2, 8
    q = Tensor(np.random.uniform(-1.0, 1.0, (tdim, heads, dim)).astype(
        np.float16), ms.float16)
    k = Tensor(np.random.uniform(-1.0, 1.0, (tdim, heads, dim)).astype(
        np.float16), ms.float16)
    v = Tensor(np.random.uniform(-1.0, 1.0, (tdim, heads, dim)).astype(
        np.float16), ms.float16)
    aseq_q = Tensor(np.array([16, 32], dtype=np.int64), ms.int64)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, heads, actual_seq_qlen=aseq_q,
                      input_layout='TND', sparse_mode=0)
        _pynative_executor.sync()


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_tnd_actual_seq_last_not_equal_t_raises(mode):
    """
    Feature: flash_attention_score
    Description: TND layout requires actual_seq_qlen/actual_seq_kvlen
                  last number equal to T
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    tdim, heads, dim = 32, 2, 8
    q = Tensor(np.random.uniform(-1.0, 1.0, (tdim, heads, dim)).astype(
        np.float16), ms.float16)
    k = Tensor(np.random.uniform(-1.0, 1.0, (tdim, heads, dim)).astype(
        np.float16), ms.float16)
    v = Tensor(np.random.uniform(-1.0, 1.0, (tdim, heads, dim)).astype(
        np.float16), ms.float16)
    # T=32, but last number is 30, should raise error
    aseq = Tensor(np.array([16, 30], dtype=np.int64), ms.int64)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, heads, actual_seq_qlen=aseq,
                      actual_seq_kvlen=aseq, input_layout='TND', sparse_mode=0)
        _pynative_executor.sync()


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=0.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_keep_prob_out_of_range_raises(mode):
    """
    Feature: flash_attention_score
    Description: keep_prob not in (0, 1] should raise at runtime
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    q, k, v = _gen_bsh_inputs()
    # keep_prob = 0.0 or >1.0 is invalid
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, 4, keep_prob=1.5, input_layout='BSH',
                      sparse_mode=0)
        _pynative_executor.sync()


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=0.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_bsh_hidden_not_divisible_by_headnum_raises(mode):
    """
    Feature: flash_attention_score
    Description: BSH hidden size not divisible by head_num should raise
                  at runtime
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    # Hidden = 30, head_num = 8 -> not divisible
    q = Tensor(np.random.uniform(-1.0, 1.0, (2, 16, 30)).astype(np.float16),
               ms.float16)
    k = Tensor(np.random.uniform(-1.0, 1.0, (2, 16, 30)).astype(np.float16),
               ms.float16)
    v = Tensor(np.random.uniform(-1.0, 1.0, (2, 16, 30)).astype(np.float16),
               ms.float16)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, 8, input_layout='BSH', sparse_mode=0)
        _pynative_executor.sync()


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential',
    mem_peak=0.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_bnsd_query_headnum_mismatch_raises(mode):
    """
    Feature: flash_attention_score
    Description: BNSD layout with query heads not equal to head_num
                  should raise at runtime
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    # query heads = 4 but head_num = 8
    q = Tensor(np.random.uniform(-1.0, 1.0, (2, 4, 16, 8)).astype(np.float16),
               ms.float16)
    k = Tensor(np.random.uniform(-1.0, 1.0, (2, 2, 16, 8)).astype(np.float16),
               ms.float16)
    v = Tensor(np.random.uniform(-1.0, 1.0, (2, 2, 16, 8)).astype(np.float16),
               ms.float16)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, 8, input_layout='BNSD', sparse_mode=0)
        _pynative_executor.sync()


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential',
    mem_peak=0.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_sbh_wrong_rank_raises(mode):
    """
    Feature: flash_attention_score
    Description: SBH layout requires 3D inputs; providing wrong rank
                  should raise at runtime
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    # Provide 2D tensors instead of [S, B, H]
    q = Tensor(np.random.uniform(-1.0, 1.0, (16, 32)).astype(np.float16),
               ms.float16)
    k = Tensor(np.random.uniform(-1.0, 1.0, (16, 32)).astype(np.float16),
               ms.float16)
    v = Tensor(np.random.uniform(-1.0, 1.0, (16, 32)).astype(np.float16),
               ms.float16)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, 2, input_layout='SBH', sparse_mode=0)
        _pynative_executor.sync()


def _gen_bsh_mismatch_heads(b=2, s=16, n_q=4, n_kv=3, d=16):
    """Generate BSH inputs with different heads for Q and KV."""
    q = np.random.uniform(-1.0, 1.0, (b, s, n_q * d)).astype(np.float16)
    k = np.random.uniform(-1.0, 1.0, (b, s, n_kv * d)).astype(np.float16)
    v = np.random.uniform(-1.0, 1.0, (b, s, n_kv * d)).astype(np.float16)
    return Tensor(q, ms.float16), Tensor(k, ms.float16), Tensor(v, ms.float16)


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=1.00
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_n2_not_factor_of_n1_raises(mode):
    """
    Feature: flash_attention_score
    Description: KV head num must be a factor of Q head num; otherwise
                  raise at runtime
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    # N1=4, N2=3 -> not a factor
    q, k, v = _gen_bsh_mismatch_heads(n_q=4, n_kv=3, d=16)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, 4, input_layout='BSH', sparse_mode=0)
        _pynative_executor.sync()


def _gen_tnd_inputs(b=2, n_q=2, n_kv=2, s1=8, s2=8, d=16):
    """Generate TND layout inputs with given parameters."""
    t1 = b * s1
    t2 = b * s2
    q = Tensor(np.random.uniform(-1.0, 1.0, (t1, n_q, d)).astype(
        np.float16), ms.float16)
    k = Tensor(np.random.uniform(-1.0, 1.0, (t2, n_kv, d)).astype(
        np.float16), ms.float16)
    v = Tensor(np.random.uniform(-1.0, 1.0, (t2, n_kv, d)).astype(
        np.float16), ms.float16)
    return q, k, v


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_tnd_attn_mask_wrong_shape_raises(mode):
    """
    Feature: flash_attention_score
    Description: TND requires attn_mask to be (2048, 2048) when provided
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    q, k, v = _gen_tnd_inputs()
    attn_mask = Tensor(np.ones((16, 16), dtype=np.uint8))
    aseq = Tensor(np.array([8, 16], dtype=np.int64), ms.int64)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, 2, attn_mask=attn_mask, actual_seq_qlen=aseq,
                      actual_seq_kvlen=aseq, input_layout='TND', sparse_mode=3)
        _pynative_executor.sync()


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=2.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_tnd_actual_seq_len_mismatch_raises(mode):
    """
    Feature: flash_attention_score
    Description: TND requires actual_seq_qlen and actual_seq_kvlen length equal
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    q, k, v = _gen_tnd_inputs()
    aseq_q = Tensor(np.array([8, 16], dtype=np.int64), ms.int64)
    aseq_k = Tensor(np.array([8], dtype=np.int64), ms.int64)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, 2, actual_seq_qlen=aseq_q,
                      actual_seq_kvlen=aseq_k, input_layout='TND',
                      sparse_mode=3)
        _pynative_executor.sync()

@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_tnd_actual_seq_not_monotonic_raises(mode):
    """
    Feature: flash_attention_score
    Description: TND requires cumulative non-decreasing sequences for
                  actual_seq lists
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    q, k, v = _gen_tnd_inputs()
    aseq = Tensor(np.array([8, 7], dtype=np.int64), ms.int64)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, 2, actual_seq_qlen=aseq, actual_seq_kvlen=aseq,
                      input_layout='TND', sparse_mode=3)
        _pynative_executor.sync()


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_tnd_real_shift_requires_same_seq_lists_raises(mode):
    """
    Feature: flash_attention_score
    Description: In TND, when real_shift is provided, two actual_seq lists
                  must be identical
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    q, k, v = _gen_tnd_inputs()
    real_shift = Tensor(np.random.uniform(-1.0, 1.0, (1, 2, 8, 8)).astype(
        np.float16), ms.float16)
    aseq_q = Tensor(np.array([8, 16], dtype=np.int64), ms.int64)
    aseq_k = Tensor(np.array([10, 16], dtype=np.int64), ms.int64)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, 2, real_shift=real_shift,
                      actual_seq_qlen=aseq_q, actual_seq_kvlen=aseq_k,
                      input_layout='TND', sparse_mode=3)
        _pynative_executor.sync()


@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=2.01
)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_attn_mask_wrong_dtype_raises(mode):
    """
    Feature: flash_attention_score
    Description: Attn mask wrong dtype should raise at runtime
    Expectation: raise RuntimeError
    """
    set_mode(mode)
    q, k, v = _gen_bsh_inputs()
    attn_mask = Tensor(np.ones((16, 16), dtype=np.float16), ms.float16)
    with pytest.raises(RuntimeError):
        _ = _fas_call(q, k, v, 4, attn_mask=attn_mask, input_layout='BSH',
                      sparse_mode=0)
        _pynative_executor.sync()

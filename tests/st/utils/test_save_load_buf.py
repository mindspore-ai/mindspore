"""Module test for save_tensor_to_buf / load_tensor_from_buf"""
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

import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import shared_memory

import numpy as np
import pytest
import mindspore as ms

load_tensor_from_buf = ms._load_tensor_from_buf  # pylint: disable=protected-access
save_tensor_to_buf = ms._save_tensor_to_buf  # pylint: disable=protected-access

ms.set_context(device_target="CPU")

DTYPES = [ms.float32, ms.float16, ms.bfloat16, ms.int32, ms.int64]
DTYPE_NP = {ms.float32: np.float32, ms.float16: np.float16, ms.int32: np.int32, ms.int64: np.int64}
DTYPE_ITEMSIZE = {ms.float32: 4, ms.float16: 2, ms.bfloat16: 2, ms.int32: 4, ms.int64: 8}

SHAPES_SMALL = [
    (1,),
    (7,),
    (3, 4),
    (2, 3, 4),
]

SHAPES_MEDIUM = [
    (32, 32),
    (256, 256),
    (1024, 1024),
]

SHAPES_LARGE = [
    (1024, 1024, 16),     # 16M elements
    (1024, 1024, 64),     # 64M elements
]

SHAPES_8GB = {
    ms.float32:  (2, 1024, 1024, 1024),
    ms.float16:  (4, 1024, 1024, 1024),
    ms.bfloat16: (4, 1024, 1024, 1024),
    ms.int32:    (2, 1024, 1024, 1024),
    ms.int64:    (1, 1024, 1024, 1024),
}


def _save_tensor_to_buf(buf, tensor, meta):
    save_tensor_to_buf(tensor, buf, meta["dtype"], meta["shape"])


def _load_tensor_from_buf(buf, meta):
    return load_tensor_from_buf(buf, meta["dtype"], meta["shape"])


def _make_tensor(dtype, shape):
    if dtype in DTYPE_NP:
        return ms.Tensor(np.random.randn(*shape).astype(DTYPE_NP[dtype]))
    return ms.Tensor(np.random.randn(*shape).astype(np.float32)).astype(dtype)


def _make_large_tensor(dtype, shape):
    numel = int(np.prod(shape))
    pattern_size = 1024 * 1024
    if dtype in DTYPE_NP:
        pattern = np.random.randn(pattern_size).astype(DTYPE_NP[dtype])
    else:
        pattern = np.random.randn(pattern_size).astype(np.float32)
    flat_np = np.tile(pattern, (numel + pattern_size - 1) // pattern_size)[:numel]
    tensor = ms.Tensor(flat_np.reshape(shape))
    if dtype == ms.bfloat16:
        tensor = tensor.astype(ms.bfloat16)
    del flat_np
    return tensor


def _verify_data(tensor, loaded, dtype):
    if dtype == ms.bfloat16:
        return np.array_equal(tensor.float().asnumpy(), loaded.float().asnumpy())
    return np.array_equal(tensor.asnumpy(), loaded.asnumpy())


def _spot_check(tensor, loaded, dtype):
    numel = int(np.prod(tensor.shape))
    flat_orig = tensor.reshape((-1,))
    flat_load = loaded.reshape((-1,))
    for idx in [0, numel // 4, numel // 2, 3 * numel // 4, numel - 1]:
        if dtype == ms.bfloat16:
            o = float(flat_orig[idx].float().asnumpy())
            l = float(flat_load[idx].float().asnumpy())
        else:
            o = float(flat_orig[idx].asnumpy())
            l = float(flat_load[idx].asnumpy())
        assert o == l, f"data mismatch at index {idx}: {o} vs {l}"


def _run_round_trip(dtype, shape, use_large=False):
    numel = int(np.prod(shape))
    nbytes = numel * DTYPE_ITEMSIZE[dtype]
    meta = {"dtype": dtype, "shape": shape, "numel": numel}
    tensor = _make_large_tensor(dtype, shape) if use_large else _make_tensor(dtype, shape)
    buf = bytearray(nbytes)

    _save_tensor_to_buf(buf, tensor, meta)
    loaded = _load_tensor_from_buf(buf, meta)

    assert loaded.shape == tensor.shape
    assert loaded.dtype == tensor.dtype

    if use_large:
        _spot_check(tensor, loaded, dtype)
    else:
        assert _verify_data(tensor, loaded, dtype)

    del tensor, loaded, buf


# ==================== small shapes x all dtypes ====================
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES_SMALL, ids=str)
def test_small_shapes(dtype, shape):
    """
    Feature: Tensor buffer save and load.
    Description: Round-trip each supported dtype with a small Tensor shape.
    Expectation: The loaded Tensor has the same shape, dtype, and values.
    """
    _run_round_trip(dtype, shape)


# ==================== medium shapes x all dtypes ====================
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES_MEDIUM, ids=str)
def test_medium_shapes(dtype, shape):
    """
    Feature: Tensor buffer save and load.
    Description: Round-trip each supported dtype with a medium Tensor shape.
    Expectation: The loaded Tensor has the same shape, dtype, and values.
    """
    _run_round_trip(dtype, shape)


# ==================== large shapes x all dtypes ====================
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES_LARGE, ids=str)
def test_large_shapes(dtype, shape):
    """
    Feature: Tensor buffer save and load.
    Description: Round-trip each supported dtype with a large Tensor shape.
    Expectation: The loaded Tensor has the same shape, dtype, and sampled values.
    """
    _run_round_trip(dtype, shape)


# ==================== 8GB x all dtypes ====================
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_8gb(dtype):
    """
    Feature: Large Tensor buffer save and load.
    Description: Round-trip an 8 GiB Tensor for each supported dtype.
    Expectation: The loaded Tensor has the same shape, dtype, and sampled values.
    """
    shape = SHAPES_8GB[dtype]
    _run_round_trip(dtype, shape, use_large=True)
    gc.collect()


# ==================== non-tensor input ====================
def test_save_rejects_non_tensor_input():
    """
    Feature: Tensor buffer input validation.
    Description: Call the save API with a string instead of a MindSpore Tensor.
    Expectation: A TypeError is raised before the destination buffer is modified.
    """
    buf = bytearray(16)
    original = bytes(buf)

    with pytest.raises(TypeError, match="tensor must be a MindSpore Tensor"):
        save_tensor_to_buf("not_a_tensor", buf, ms.float32, (4,))

    assert buf == original


# ==================== shared memory x all dtypes ====================
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_shared_memory(dtype):
    """
    Feature: Shared-memory Tensor buffer save and load.
    Description: Round-trip each supported dtype through a shared-memory buffer.
    Expectation: The loaded Tensor contains the original values.
    """
    shape = (64, 64)
    numel = 64 * 64
    nbytes = numel * DTYPE_ITEMSIZE[dtype]
    meta = {"dtype": dtype, "shape": shape, "numel": numel}
    tensor = _make_tensor(dtype, shape)

    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    try:
        _save_tensor_to_buf(shm.buf, tensor, meta)
        loaded = _load_tensor_from_buf(shm.buf, meta)
        assert _verify_data(tensor, loaded, dtype)
    finally:
        shm.close()
        shm.unlink()


# ============ Simulated model weights for multi-thread tests ============
# Mimics a typical transformer-like model: embedding, attention, FFN, layernorm
MODEL_WEIGHTS_SMALL = [
    # (name, dtype, shape)
    ("embed_tokens", ms.float32, (256, 64)),
    ("attn.q_proj", ms.float16, (64, 64)),
    ("attn.k_proj", ms.float16, (64, 64)),
    ("attn.v_proj", ms.float16, (64, 64)),
    ("attn.o_proj", ms.float16, (64, 64)),
    ("ffn.gate_proj", ms.bfloat16, (64, 256)),
    ("ffn.up_proj", ms.bfloat16, (64, 256)),
    ("ffn.down_proj", ms.bfloat16, (256, 64)),
    ("layernorm.weight", ms.float32, (64,)),
    ("layernorm.bias", ms.float32, (64,)),
]

MODEL_WEIGHTS_LARGE = [
    # Simulates larger model layers (~100MB+ per weight)
    ("embed_tokens", ms.float32, (32000, 4096)),       # ~500MB
    ("layer0.attn.q_proj", ms.float16, (4096, 4096)),  # ~32MB
    ("layer0.attn.k_proj", ms.float16, (4096, 4096)),
    ("layer0.attn.v_proj", ms.float16, (4096, 4096)),
    ("layer0.attn.o_proj", ms.float16, (4096, 4096)),
    ("layer0.ffn.gate_proj", ms.bfloat16, (4096, 11008)),  # ~86MB
    ("layer0.ffn.up_proj", ms.bfloat16, (4096, 11008)),
    ("layer0.ffn.down_proj", ms.bfloat16, (11008, 4096)),
    ("layer0.layernorm.weight", ms.float32, (4096,)),
    ("layer0.layernorm.bias", ms.float32, (4096,)),
]


def _build_model_state(weight_defs):
    """Build model state dict: {name: (tensor, buf, meta)}."""
    state = {}
    for name, dtype, shape in weight_defs:
        tensor = _make_tensor(dtype, shape)
        numel = int(np.prod(shape))
        nbytes = numel * DTYPE_ITEMSIZE[dtype]
        buf = bytearray(nbytes)
        meta = {"dtype": dtype, "shape": shape, "numel": numel}
        state[name] = (tensor, buf, meta)
    return state


def _save_one_weight(args):
    """Thread target: save a single weight tensor to its buffer."""
    name, tensor, buf, meta = args
    _save_tensor_to_buf(buf, tensor, meta)
    return name


# ==================== multi-thread save: small model ====================
@pytest.mark.parametrize("num_threads", [2, 4, 8], ids=lambda n: f"{n}threads")
def test_multithread_save_small_model(num_threads):
    """
    Feature: Concurrent Tensor buffer save.
    Description: Save small-model weights concurrently with two, four, or eight Python threads.
    Expectation: Every saved weight can be loaded with its original shape, dtype, and values.
    """
    state = _build_model_state(MODEL_WEIGHTS_SMALL)

    tasks = [(name, tensor, buf, meta) for name, (tensor, buf, meta) in state.items()]
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(_save_one_weight, t) for t in tasks]
        for f in as_completed(futures):
            f.result()  # raise if any thread failed

    for name, (tensor, buf, meta) in state.items():
        loaded = _load_tensor_from_buf(buf, meta)
        assert loaded.shape == tensor.shape
        assert loaded.dtype == tensor.dtype
        assert _verify_data(tensor, loaded, meta["dtype"]), f"data mismatch for {name}"


# ==================== multi-thread save: large model ====================
@pytest.mark.parametrize("num_threads", [2, 4, 8], ids=lambda n: f"{n}threads")
def test_multithread_save_large_model(num_threads):
    """
    Feature: Concurrent large Tensor buffer save.
    Description: Save large-model weights concurrently with two, four, or eight Python threads.
    Expectation: Every saved weight can be loaded with its original shape, dtype, and values.
    """
    state = _build_model_state(MODEL_WEIGHTS_LARGE)

    tasks = [(name, tensor, buf, meta) for name, (tensor, buf, meta) in state.items()]
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(_save_one_weight, t) for t in tasks]
        for f in as_completed(futures):
            f.result()

    for name, (tensor, buf, meta) in state.items():
        loaded = _load_tensor_from_buf(buf, meta)
        assert loaded.shape == tensor.shape
        assert loaded.dtype == tensor.dtype
        assert _verify_data(tensor, loaded, meta["dtype"]), f"data mismatch for {name}"


# ==================== multi-thread save: same dtype, different sizes ====================
def test_multithread_save_same_dtype_varied_shapes():
    """
    Feature: Concurrent Tensor buffer save with varied shapes.
    Description: Save float32 weights of varied shapes concurrently with eight threads.
    Expectation: Every saved weight can be loaded with its original values.
    """
    weight_defs = [
        ("w_tiny", ms.float32, (1,)),
        ("w_small", ms.float32, (16, 16)),
        ("w_medium", ms.float32, (512, 512)),
        ("w_large", ms.float32, (2048, 2048)),
        ("w_xlarge", ms.float32, (4096, 4096)),
    ]
    state = _build_model_state(weight_defs)

    tasks = [(name, tensor, buf, meta) for name, (tensor, buf, meta) in state.items()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_save_one_weight, t) for t in tasks]
        for f in as_completed(futures):
            f.result()

    for name, (tensor, buf, meta) in state.items():
        loaded = _load_tensor_from_buf(buf, meta)
        assert _verify_data(tensor, loaded, meta["dtype"]), f"data mismatch for {name}"


# ==================== multi-thread save + load concurrently ====================
def test_multithread_save_and_load_concurrent():
    """
    Feature: Concurrent Tensor buffer save and load.
    Description: Save even-indexed weights while loading odd-indexed weights concurrently.
    Expectation: All concurrent calls finish and every buffer retains the expected values.
    """
    weight_defs = [
        (f"layer{i}.weight", ms.float32, (512, 512)) for i in range(8)
    ]
    state = _build_model_state(weight_defs)

    # First save all weights sequentially to populate buffers
    for name, (tensor, buf, meta) in state.items():
        _save_tensor_to_buf(buf, tensor, meta)

    # Now concurrently: re-save even layers, load odd layers
    errors = []

    def save_task(name, tensor, buf, meta):
        _save_tensor_to_buf(buf, tensor, meta)

    def load_task(name, buf, meta, expected_tensor):
        loaded = _load_tensor_from_buf(buf, meta)
        if not _verify_data(expected_tensor, loaded, meta["dtype"]):
            errors.append(f"load mismatch for {name}")

    names = list(state.keys())
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for i, name in enumerate(names):
            tensor, buf, meta = state[name]
            if i % 2 == 0:
                futures.append(pool.submit(save_task, name, tensor, buf, meta))
            else:
                futures.append(pool.submit(load_task, name, buf, meta, tensor))
        for f in as_completed(futures):
            f.result()

    assert not errors, f"Errors: {errors}"

    # Verify all weights after concurrent operations
    for name, (tensor, buf, meta) in state.items():
        loaded = _load_tensor_from_buf(buf, meta)
        assert _verify_data(tensor, loaded, meta["dtype"]), f"final verify failed for {name}"


# ==================== multi-thread save with shared memory ====================
def test_multithread_save_shared_memory():
    """
    Feature: Concurrent Tensor save to shared memory.
    Description: Save small-model weights concurrently into independent shared-memory buffers.
    Expectation: Every shared-memory buffer can be loaded with the original values.
    """
    weight_defs = MODEL_WEIGHTS_SMALL
    shm_list = []
    state = {}

    try:
        for name, dtype, shape in weight_defs:
            tensor = _make_tensor(dtype, shape)
            numel = int(np.prod(shape))
            nbytes = numel * DTYPE_ITEMSIZE[dtype]
            meta = {"dtype": dtype, "shape": shape, "numel": numel}
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            shm_list.append(shm)
            state[name] = (tensor, shm.buf, meta)

        tasks = [(name, tensor, buf, meta) for name, (tensor, buf, meta) in state.items()]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_save_one_weight, t) for t in tasks]
            for f in as_completed(futures):
                f.result()

        for name, (tensor, buf, meta) in state.items():
            loaded = _load_tensor_from_buf(buf, meta)
            assert _verify_data(tensor, loaded, meta["dtype"]), f"shm mismatch for {name}"
    finally:
        for shm in shm_list:
            shm.close()
            shm.unlink()


# ==================== offset tests ====================
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_offset_save_load(dtype):
    """
    Feature: Tensor buffer save and load with offsets.
    Description: Store three Tensors in non-overlapping regions of one buffer and load each region.
    Expectation: Every loaded Tensor has the original shape, dtype, and values.
    """
    shapes = [(4, 4), (8,), (2, 3)]
    itemsize = DTYPE_ITEMSIZE[dtype]

    # Calculate total buffer size and per-tensor offsets
    offsets = []
    cur_offset = 0
    for shape in shapes:
        offsets.append(cur_offset)
        cur_offset += int(np.prod(shape)) * itemsize
    total_bytes = cur_offset

    buf = bytearray(total_bytes)
    tensors = [_make_tensor(dtype, s) for s in shapes]

    # Save each tensor at its offset
    for tensor, shape, offset in zip(tensors, shapes, offsets):
        numel = int(np.prod(shape))
        save_tensor_to_buf(
            tensor=tensor, buf=buf, dtype=dtype, shape=shape, count=numel, offset=offset
        )

    # Load each tensor back from its offset
    for tensor, shape, offset in zip(tensors, shapes, offsets):
        numel = int(np.prod(shape))
        loaded = load_tensor_from_buf(buf=buf, dtype=dtype, shape=shape, count=numel, offset=offset)
        assert loaded.shape == tensor.shape
        assert loaded.dtype == tensor.dtype
        assert _verify_data(tensor, loaded, dtype), f"offset={offset} data mismatch"


def test_offset_kwargs():
    """
    Feature: Tensor buffer API calling conventions.
    Description: Exercise all-keyword, mixed-argument, and default-argument calls.
    Expectation: Every calling convention round-trips the original Tensor values.
    """
    dtype = ms.float32
    shape = (3, 4)
    numel = 12
    itemsize = 4
    offset = 16  # skip 16 bytes at start
    total = offset + numel * itemsize

    tensor = _make_tensor(dtype, shape)
    buf = bytearray(total)

    # All kwargs
    save_tensor_to_buf(tensor=tensor, buf=buf, dtype=dtype, shape=shape, count=numel, offset=offset)
    loaded = load_tensor_from_buf(buf=buf, dtype=dtype, shape=shape, count=numel, offset=offset)
    assert _verify_data(tensor, loaded, dtype)

    # count/offset as kwargs, rest positional
    buf2 = bytearray(total)
    save_tensor_to_buf(tensor, buf2, dtype, shape, count=numel, offset=offset)
    loaded2 = load_tensor_from_buf(buf2, dtype, shape, count=numel, offset=offset)
    assert _verify_data(tensor, loaded2, dtype)

    # count/offset omitted (defaults: count=-1, offset=0)
    buf3 = bytearray(numel * itemsize)
    save_tensor_to_buf(tensor=tensor, buf=buf3, dtype=dtype, shape=shape)
    loaded3 = load_tensor_from_buf(buf=buf3, dtype=dtype, shape=shape)
    assert _verify_data(tensor, loaded3, dtype)


# ==================== load shape validation ====================
def test_load_rejects_shape_larger_than_buffer():
    """
    Feature: Tensor buffer load validation.
    Description: Load a Tensor whose shape claims more int32 elements than the 12-byte buffer provides.
    Expectation: A ValueError is raised instead of returning a Tensor that reads out of bounds.
    """
    buf = bytearray(12)  # 3 int32 elements

    with pytest.raises(ValueError, match="The buffer holds 3 elements"):
        load_tensor_from_buf(buf, ms.int32, (1000,))


def test_load_rejects_shape_not_matching_explicit_count():
    """
    Feature: Tensor buffer load validation.
    Description: Load a Tensor whose shape describes fewer elements than the explicitly requested count.
    Expectation: A ValueError is raised because shape and count disagree.
    """
    buf = bytearray(48)  # 12 int32 elements

    with pytest.raises(ValueError, match="The buffer holds 12 elements"):
        load_tensor_from_buf(buf, ms.int32, (2, 2), count=12)

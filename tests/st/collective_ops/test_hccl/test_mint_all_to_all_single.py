# Copyright 2026 Huawei Technologies Co., Ltd
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

"""Two-rank contract tests for mint.distributed.all_to_all_single, launched with msrun."""

import inspect
import importlib

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
from mindspore._c_expression import TensorPy, _validate_all_to_all_single_and_build_comm_args
from mindspore.mint import distributed as dist
from mindspore.communication import GlobalComm
from mindspore.communication.comm_func import CommHandle


@pytest.fixture(name="rank", scope="module")
def initialized_rank():
    """Initialize the two participating ranks once for the communication tests."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    dist.init_process_group()
    assert dist.get_world_size() == 2, "Launch this module with msrun --worker_num=2 --local_worker_num=2."
    yield dist.get_rank()
    ms.runtime.synchronize()
    dist.destroy_process_group()


SEND_VALUES = ([0, 1, 2, 3], [10, 11, 12, 13])
RECEIVED_VALUES = ([0, 1, 10, 11], [2, 3, 12, 13])
COMPLEX_SEND_VALUES = ([1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j])
COMPLEX_RECEIVED_VALUES = ([1 + 2j, 5 + 6j], [3 + 4j, 7 + 8j])
DISPATCH_CASES = [
    pytest.param(ms.float32, SEND_VALUES, RECEIVED_VALUES, id="float32"),
    pytest.param(ms.complex64, COMPLEX_SEND_VALUES, COMPLEX_RECEIVED_VALUES, id="complex64"),
    pytest.param(ms.complex128, COMPLEX_SEND_VALUES, COMPLEX_RECEIVED_VALUES, id="complex128"),
    pytest.param(
        ms.float32,
        ([[0, 1], [2, 3]], [[10, 11], [12, 13]]),
        ([[0, 1], [10, 11]], [[2, 3], [12, 13]]),
        id="float32-matrix",
    ),
    pytest.param(
        ms.complex64,
        ([[1 + 2j], [3 + 4j]], [[5 + 6j], [7 + 8j]]),
        ([[1 + 2j], [5 + 6j]], [[3 + 4j], [7 + 8j]]),
        id="complex64-matrix",
    ),
    pytest.param(
        ms.complex128,
        ([[1 + 2j], [3 + 4j]], [[5 + 6j], [7 + 8j]]),
        ([[1 + 2j], [5 + 6j]], [[3 + 4j], [7 + 8j]]),
        id="complex128-matrix",
    ),
]


def test_public_signature_matches_pytorch():
    """
    Feature: mint.distributed.all_to_all_single signature.
    Description: Inspect the public function's parameters and their defaults.
    Expectation: Parameter names, order and defaults match PyTorch.
    """
    signature = inspect.signature(dist.all_to_all_single)
    assert list(signature.parameters) == [
        "output", "input", "output_split_sizes", "input_split_sizes", "group", "async_op"
    ]
    assert [parameter.default for parameter in list(signature.parameters.values())[2:]] == [None, None, None, False]


def test_exchange_uses_the_existing_python_communication_operator(rank, monkeypatch):
    """
    Feature: mint.distributed.all_to_all_single communication entry.
    Description: Record the standard communication call while executing the real two-rank exchange.
    Expectation: The existing operator receives the original tensors once with the prepared element counts.
    """
    module = importlib.import_module("mindspore.mint.distributed.distributed")
    original_op = module.dist_comm_all_to_all_v_single_op
    calls = []

    def record_communication(*args):
        calls.append(args)
        return original_op(*args)

    monkeypatch.setattr(module, "dist_comm_all_to_all_v_single_op", record_communication)
    input_tensor = Tensor(([[0, 1], [2, 3]], [[10, 11], [12, 13]])[rank], ms.float32)
    output = Tensor([[-1, -1], [-1, -1]], ms.float32)

    result = dist.all_to_all_single(output, input_tensor)

    assert result is None
    assert len(calls) == 1
    assert calls[0][0] is output
    assert calls[0][1] is input_tensor
    assert calls[0][3:] == ((2, 2), (2, 2), 2, True, False)
    expected = ([[0, 1], [10, 11]], [[2, 3], [12, 13]])
    np.testing.assert_array_equal(output.asnumpy(), expected[rank])


@pytest.mark.parametrize("splits", [
    pytest.param(None, id="default"),
    pytest.param([], id="empty-list"),
    pytest.param((), id="empty-tuple"),
    pytest.param([2, 2], id="explicit-list"),
    pytest.param((2, 2), id="explicit-tuple"),
])
def test_equal_splits_write_existing_output_and_return_none(rank, splits):
    """
    Feature: mint.distributed.all_to_all_single equal splits.
    Description: Exchange four values with omitted, empty or explicit equal splits.
    Expectation: The call returns None and writes the expected values into the original output storage.
    """
    input_tensor = Tensor(SEND_VALUES[rank], ms.float32)
    output = Tensor([-1, -1, -1, -1], ms.float32).to("Ascend")
    output_ptr = output.data_ptr()

    result = dist.all_to_all_single(output, input_tensor, splits, splits)

    assert result is None
    assert output.data_ptr() == output_ptr
    assert output.shape == (4,)
    assert output.dtype == ms.float32
    np.testing.assert_array_equal(output.asnumpy(), RECEIVED_VALUES[rank])


@pytest.mark.parametrize("dtype", [ms.int32, ms.int64, ms.float16, ms.float32, ms.bfloat16],
                         ids=["int32", "int64", "float16", "float32", "bfloat16"])
def test_equal_splits_preserve_values_and_dtype(rank, dtype):
    """
    Feature: mint.distributed.all_to_all_single numeric dtypes.
    Description: Exchange equal splits of integer and floating-point tensors.
    Expectation: The output preserves the requested dtype and contains the expected values.
    """
    input_tensor = Tensor(SEND_VALUES[rank], dtype)
    output = Tensor([-1, -1, -1, -1], dtype)

    dist.all_to_all_single(output, input_tensor)

    assert output.dtype == dtype
    np.testing.assert_array_equal(output.float().asnumpy(), RECEIVED_VALUES[rank])


def test_uneven_row_splits_allow_different_input_and_output_lengths(rank):
    """
    Feature: mint.distributed.all_to_all_single uneven splits.
    Description: Exchange two-column tensors with unequal row splits and different input and output lengths.
    Expectation: Splits count rows and each output contains the expected peer rows.
    """
    inputs = ([[0, 1], [2, 3], [4, 5]], [[10, 11], [12, 13], [14, 15], [16, 17]])
    input_splits = ([1, 2], [3, 1])
    output_splits = ([1, 3], [2, 1])
    expected = ([[0, 1], [10, 11], [12, 13], [14, 15]], [[2, 3], [4, 5], [16, 17]])
    input_tensor = Tensor(inputs[rank], ms.float32)
    output = Tensor(np.full((4 if rank == 0 else 3, 2), -1, np.float32))

    dist.all_to_all_single(output, input_tensor, output_splits[rank], input_splits[rank])

    np.testing.assert_array_equal(output.asnumpy(), expected[rank])


def test_omitted_input_splits_are_inferred_independently(rank):
    """
    Feature: mint.distributed.all_to_all_single omitted input splits.
    Description: Omit input splits for senders with different input lengths and provide output splits.
    Expectation: Each input is split equally using its own length and both outputs contain the expected values.
    """
    inputs = ([0, 1, 2, 3], [10, 11, 12, 13, 14, 15])
    expected = ([0, 1, 10, 11, 12], [2, 3, 13, 14, 15])
    output = Tensor(np.full(5, -1, np.float32))

    dist.all_to_all_single(output, Tensor(inputs[rank], ms.float32), output_split_sizes=[2, 3])

    np.testing.assert_array_equal(output.asnumpy(), expected[rank])


def test_omitted_output_splits_are_inferred_independently(rank):
    """
    Feature: mint.distributed.all_to_all_single omitted output splits.
    Description: Provide uneven input splits and omit output splits for receive buffers of different lengths.
    Expectation: Each output infers its own equal splits and receives the expected values.
    """
    inputs = ([0, 1, 2, 3, 4], [10, 11, 12, 13, 14])
    expected = ([0, 1, 10, 11], [2, 3, 4, 12, 13, 14])
    output = Tensor(np.full(4 if rank == 0 else 6, -1, np.float32))

    dist.all_to_all_single(output, Tensor(inputs[rank], ms.float32), input_split_sizes=[2, 3])

    np.testing.assert_array_equal(output.asnumpy(), expected[rank])


@pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("dtype", [ms.complex64, ms.complex128], ids=["complex64", "complex128"])
def test_complex_exchange_preserves_real_and_imaginary_components(rank, dtype, async_op):
    """
    Feature: mint.distributed.all_to_all_single complex tensors.
    Description: Exchange complex vectors synchronously and asynchronously, waiting on any returned handle.
    Expectation: Both components arrive correctly while the output storage, shape and dtype remain unchanged.
    """
    inputs = ([1 + 11j, 2 + 12j, 3 + 13j, 4 + 14j], [5 + 15j, 6 + 16j, 7 + 17j, 8 + 18j])
    expected = ([1 + 11j, 2 + 12j, 5 + 15j, 6 + 16j], [3 + 13j, 4 + 14j, 7 + 17j, 8 + 18j])
    input_tensor = Tensor(inputs[rank], dtype)
    output = Tensor([-1j, -1j, -1j, -1j], dtype).to("Ascend")
    output_ptr = output.data_ptr()

    handle = dist.all_to_all_single(output, input_tensor, async_op=async_op)
    if async_op:
        assert isinstance(handle, CommHandle)
        handle.wait()
    else:
        assert handle is None

    assert output.data_ptr() == output_ptr
    assert output.shape == (4,)
    assert output.dtype == dtype
    np.testing.assert_array_equal(output.asnumpy(), expected[rank])


@pytest.mark.parametrize("dtype", [ms.complex64, ms.complex128], ids=["complex64", "complex128"])
def test_complex_uneven_splits_count_original_rows(rank, dtype):
    """
    Feature: mint.distributed.all_to_all_single complex row splits.
    Description: Exchange complex matrices using uneven row splits that include an empty split.
    Expectation: Splits count original complex rows and preserve both components and the output dtype.
    """
    inputs = ([[1 + 11j, 2 + 12j]], [[3 + 13j, 4 + 14j], [5 + 15j, 6 + 16j]])
    input_splits = ([0, 1], [1, 1])
    output_splits = ([0, 1], [1, 1])
    expected = ([[3 + 13j, 4 + 14j]], [[1 + 11j, 2 + 12j], [5 + 15j, 6 + 16j]])
    output = Tensor(np.full((1 if rank == 0 else 2, 2), -1j), dtype)

    dist.all_to_all_single(output, Tensor(inputs[rank], dtype), output_splits[rank], input_splits[rank])

    assert output.dtype == dtype
    np.testing.assert_array_equal(output.asnumpy(), expected[rank])


def test_zero_splits_exchange_only_the_nonempty_peer_chunks(rank):
    """
    Feature: mint.distributed.all_to_all_single zero splits.
    Description: Each rank sends zero elements to itself and two elements to its peer.
    Expectation: Both ranks participate and receive only their peer's values.
    """
    inputs = ([0, 1], [10, 11])
    splits = ([0, 2], [2, 0])
    expected = ([10, 11], [0, 1])
    output = Tensor([-1, -1], ms.int32)

    dist.all_to_all_single(output, Tensor(inputs[rank], ms.int32), splits[rank], splits[rank])

    np.testing.assert_array_equal(output.asnumpy(), expected[rank])


def test_rank_with_empty_input_still_receives_peer_data(rank):
    """
    Feature: mint.distributed.all_to_all_single empty sender.
    Description: Exchange data when one rank has an empty input and the other sends two values to each rank.
    Expectation: Both ranks receive the expected nonempty peer split.
    """
    inputs = ([], [10, 11, 12, 13])
    input_splits = ([0, 0], [2, 2])
    output = Tensor([-1, -1], ms.float32)
    expected = ([10, 11], [12, 13])

    dist.all_to_all_single(output, Tensor(inputs[rank], ms.float32), [0, 2], input_splits[rank])

    np.testing.assert_array_equal(output.asnumpy(), expected[rank])


@pytest.mark.parametrize("shape", [
    pytest.param((0,), id="empty-vector"),
    pytest.param((0, 3), id="zero-rows"),
    pytest.param((2, 0), id="zero-columns"),
    pytest.param((2, 0, 3), id="zero-middle-dimension"),
    pytest.param((2, 3, 0), id="zero-last-dimension"),
])
@pytest.mark.parametrize("async_op", [False, True], ids=["sync", "async"])
def test_empty_tensors_complete_without_changing_shape(rank, shape, async_op):
    """
    Feature: mint.distributed.all_to_all_single empty tensors.
    Description: Exchange tensors with a zero dimension synchronously and asynchronously.
    Expectation: The call or handle completes and the output retains its empty shape.
    """
    del rank
    input_tensor = Tensor(np.empty(shape, np.float32))
    output = Tensor(np.empty(shape, np.float32))

    handle = dist.all_to_all_single(output, input_tensor, async_op=async_op)
    if async_op:
        handle.wait()
    else:
        assert handle is None

    assert output.shape == shape
    assert output.asnumpy().size == 0


def test_contiguous_views_write_back_at_their_storage_offsets(rank):
    """
    Feature: mint.distributed.all_to_all_single contiguous views.
    Description: Exchange contiguous slices with nonzero storage offsets and sentinel values on both sides.
    Expectation: The output slice receives the expected values without replacing storage or modifying sentinels.
    """
    input_storage = Tensor(([-90, 0, 1, 2, 3, -91], [-90, 10, 11, 12, 13, -91])[rank], ms.float32)
    output_storage = Tensor([-99, -1, -1, -1, -1, -98], ms.float32).to("Ascend")
    input_view = input_storage[1:5]
    output_view = output_storage[1:5]
    assert input_view.is_contiguous() and output_view.is_contiguous()
    assert input_view.storage_offset() == output_view.storage_offset() == 1
    output_ptr = output_view.data_ptr()
    expected = ([-99, 0, 1, 10, 11, -98], [-99, 2, 3, 12, 13, -98])

    dist.all_to_all_single(output_view, input_view)

    assert output_view.data_ptr() == output_ptr
    np.testing.assert_array_equal(output_storage.asnumpy(), expected[rank])


@pytest.mark.parametrize("dtype", [ms.complex64, ms.complex128], ids=["complex64", "complex128"])
def test_complex_views_preserve_storage_offsets_and_aliases(rank, dtype):
    """
    Feature: mint.distributed.all_to_all_single complex views.
    Description: Exchange complex slices with nonzero storage offsets and surrounding sentinel values.
    Expectation: Values are written into the original slice with its offset, dtype and adjacent storage preserved.
    """
    inputs = ([-90j, 1 + 11j, 2 + 12j, -91j], [-90j, 3 + 13j, 4 + 14j, -91j])
    input_storage = Tensor(inputs[rank], dtype)
    output_storage = Tensor([-99j, -1j, -1j, -98j], dtype).to("Ascend")
    input_view = input_storage[1:3]
    output_view = output_storage[1:3]
    output_ptr = output_view.data_ptr()
    expected = ([-99j, 1 + 11j, 3 + 13j, -98j], [-99j, 2 + 12j, 4 + 14j, -98j])

    dist.all_to_all_single(output_view, input_view)

    assert output_view.storage_offset() == 1
    assert output_view.data_ptr() == output_ptr
    assert output_view.dtype == dtype
    np.testing.assert_array_equal(output_storage.asnumpy(), expected[rank])


def test_async_handle_wait_orders_computation_on_another_stream(rank):
    """
    Feature: mint.distributed.all_to_all_single asynchronous stream dependencies.
    Description: Submit on one stream, wait twice on a second stream, then compute from the received values.
    Expectation: A CommHandle is returned and the consumer observes the completed communication result.
    """
    input_tensor = Tensor(SEND_VALUES[rank], ms.float32)
    output = Tensor([-1, -1, -1, -1], ms.float32)
    producer = ms.runtime.Stream()
    consumer = ms.runtime.Stream()
    ms.runtime.synchronize()

    with ms.runtime.StreamCtx(producer):
        handle = dist.all_to_all_single(output, input_tensor, async_op=True)
    assert isinstance(handle, CommHandle)
    with ms.runtime.StreamCtx(consumer):
        handle.wait()
        handle.wait()
        result = output + 1
        actual = result.asnumpy()

    expected = ([1, 2, 11, 12], [3, 4, 13, 14])
    np.testing.assert_array_equal(actual, expected[rank])


@pytest.mark.parametrize("async_op", [False, True], ids=["sync-same-stream", "async-cross-stream"])
def test_complex_exchange_orders_real_view_computation_on_nondefault_streams(rank, async_op):
    """
    Feature: mint.distributed.all_to_all_single complex stream dependencies.
    Description: Exchange complex tensors on a nondefault stream and compute from the output's real-valued view.
    Expectation: View and communication tasks preserve both components and the consumer observes the received data.
    """
    input_tensor = Tensor(COMPLEX_SEND_VALUES[rank], ms.complex64)
    output = Tensor([-1j, -1j], ms.complex64)
    producer = ms.runtime.Stream()
    consumer = ms.runtime.Stream() if async_op else producer
    ms.runtime.synchronize()

    with ms.runtime.StreamCtx(producer):
        handle = dist.all_to_all_single(output, input_tensor, async_op=async_op)
    with ms.runtime.StreamCtx(consumer):
        if async_op:
            assert isinstance(handle, CommHandle)
            handle.wait()
        else:
            assert handle is None
        result = output.view(ms.float32) + 1
        actual = result.asnumpy()

    expected = ([2, 3, 6, 7], [4, 5, 8, 9])
    assert output.dtype == ms.complex64
    np.testing.assert_array_equal(actual, expected[rank])


@pytest.mark.parametrize("nondefault_stream", [False, True], ids=["default-stream", "nondefault-stream"])
def test_sync_exchange_orders_immediate_dependent_computation(rank, nondefault_stream):
    """
    Feature: mint.distributed.all_to_all_single synchronous stream dependencies.
    Description: Communicate between dependent operations on the default or a nondefault stream.
    Expectation: The call returns None and the immediately following computation observes the received values.
    """
    input_tensor = Tensor(SEND_VALUES[rank], ms.float32)
    output = Tensor([-1, -1, -1, -1], ms.float32)

    stream = ms.runtime.Stream() if nondefault_stream else ms.runtime.current_stream()
    ms.runtime.synchronize()

    with ms.runtime.StreamCtx(stream):
        input_tensor = input_tensor + 1
        assert dist.all_to_all_single(output, input_tensor) is None
        result = output + 1
        actual = result.asnumpy()

    expected = ([2, 3, 12, 13], [4, 5, 14, 15])
    np.testing.assert_array_equal(actual, expected[rank])


def test_explicit_group_uses_its_own_rank_count(rank):
    """
    Feature: mint.distributed.all_to_all_single named groups.
    Description: Exchange three values in a one-rank subgroup of the two-rank world group.
    Expectation: Split inference uses the subgroup size and the rank receives its own values.
    """
    group = dist.new_group([rank])
    input_tensor = Tensor([1, 2, 3], ms.float32)
    output = Tensor([-1, -1, -1], ms.float32)

    dist.all_to_all_single(output, input_tensor, group=group)

    np.testing.assert_array_equal(output.asnumpy(), [1, 2, 3])
    dist.destroy_process_group(group)


def test_omitted_group_uses_the_current_python_default_group(rank, monkeypatch):
    """
    Feature: mint.distributed.all_to_all_single default group resolution.
    Description: Temporarily select a one-rank subgroup as the default and omit the group argument.
    Expectation: Python passes the current default to preparation and exchanges all three values within that subgroup.
    """
    from mindspore.communication import create_group

    group = f"all_to_all_default_{rank}"
    create_group(group, [rank])
    input_tensor = Tensor([1, 2, 3], ms.float32)
    output = Tensor([-1, -1, -1], ms.float32)

    with monkeypatch.context() as patch:
        patch.setattr(GlobalComm, "WORLD_COMM_GROUP", group)
        dist.all_to_all_single(output, input_tensor)

    np.testing.assert_array_equal(output.asnumpy(), [1, 2, 3])
    dist.destroy_process_group(group)


def test_native_tensorpy_exchange_writes_the_supplied_output(rank):
    """
    Feature: mint.distributed.all_to_all_single TensorPy support.
    Description: Exchange native TensorPy objects without wrapping them in the Python Tensor subclass.
    Expectation: The synchronous exchange accepts both tensors and writes the received values into the supplied output.
    """
    input_tensor = TensorPy(np.array(SEND_VALUES[rank], dtype=np.float32))
    output = TensorPy(np.array([-1, -1, -1, -1], dtype=np.float32))

    result = dist.all_to_all_single(output, input_tensor)

    assert result is None
    np.testing.assert_array_equal(output.asnumpy(), RECEIVED_VALUES[rank])


@pytest.mark.parametrize("group", [
    pytest.param(1, id="integer"),
    pytest.param(b"hccl_world_group", id="bytes"),
    pytest.param([], id="list"),
])
def test_invalid_group_type_raises_type_error_before_communication(rank, group):
    """
    Feature: mint.distributed.all_to_all_single group type validation.
    Description: Supply a group value that is neither None nor a Python string.
    Expectation: The native preparation raises TypeError without submitting communication.
    """
    input_tensor = Tensor(SEND_VALUES[rank], ms.float32)
    output = Tensor([-1, -1, -1, -1], ms.float32)

    with pytest.raises(TypeError, match="group"):
        dist.all_to_all_single(output, input_tensor, group=group)

    np.testing.assert_array_equal(output.asnumpy(), [-1, -1, -1, -1])


@pytest.mark.parametrize("async_op", [
    pytest.param(0, id="integer-zero"),
    pytest.param(1, id="integer-one"),
    pytest.param(None, id="none"),
    pytest.param("True", id="string"),
    pytest.param(np.bool_(True), id="numpy-bool"),
])
def test_nonboolean_async_flag_raises_type_error_before_communication(rank, async_op):
    """
    Feature: mint.distributed.all_to_all_single asynchronous flag validation.
    Description: Supply nonboolean values that could otherwise be coerced to bool by a native binding.
    Expectation: Preparation retains the Python bool-only contract and leaves the output unchanged.
    """
    input_tensor = Tensor(SEND_VALUES[rank], ms.float32)
    output = Tensor([-1, -1, -1, -1], ms.float32)

    with pytest.raises(TypeError, match="async_op"):
        dist.all_to_all_single(output, input_tensor, async_op=async_op)

    np.testing.assert_array_equal(output.asnumpy(), [-1, -1, -1, -1])


@pytest.mark.parametrize("argument", ["input_split_sizes", "output_split_sizes"], ids=["input", "output"])
@pytest.mark.parametrize("splits,exception", [
    pytest.param(2, TypeError, id="not-a-sequence"),
    pytest.param([1.0, 3.0], TypeError, id="noninteger-elements"),
    pytest.param([-1, 5], ValueError, id="negative-size"),
    pytest.param([4], ValueError, id="wrong-length"),
    pytest.param([1, 2], ValueError, id="sum-too-small"),
    pytest.param([3, 3], ValueError, id="sum-too-large"),
])
def test_invalid_splits_raise_mint_exceptions_before_communication(rank, argument, splits, exception):
    """
    Feature: mint.distributed.all_to_all_single split validation.
    Description: Supply invalid input or output split types, elements, lengths or sums.
    Expectation: The specified TypeError or ValueError is raised and the output remains untouched.
    """
    input_tensor = Tensor(SEND_VALUES[rank], ms.float32)
    output = Tensor([-1, -1, -1, -1], ms.float32)

    with pytest.raises(exception):
        dist.all_to_all_single(output, input_tensor, **{argument: splits})

    np.testing.assert_array_equal(output.asnumpy(), [-1, -1, -1, -1])


@pytest.mark.parametrize("case", [
    "scalar", "dtype", "not-divisible", "different-numel", "input-strides", "output-strides"
])
def test_invalid_tensor_metadata_raises_value_error(rank, case):
    """
    Feature: mint.distributed.all_to_all_single tensor validation.
    Description: Supply scalar, mismatched-dtype, indivisible, unequal-numel or noncontiguous tensors.
    Expectation: Invalid metadata raises ValueError before communication is submitted.
    """
    input_tensor = Tensor(SEND_VALUES[rank], ms.float32)
    output = Tensor([-1, -1, -1, -1], ms.float32)
    if case == "scalar":
        input_tensor, output = Tensor(1.0), Tensor(0.0)
    elif case == "dtype":
        output = Tensor([-1, -1, -1, -1], ms.int32)
    elif case == "not-divisible":
        input_tensor, output = Tensor([0, 1, 2], ms.float32), Tensor([-1, -1, -1], ms.float32)
    elif case == "different-numel":
        output = Tensor([-1, -1, -1, -1, -1, -1], ms.float32)
    elif case == "input-strides":
        input_tensor = input_tensor.reshape(2, 2).transpose(1, 0)
    elif case == "output-strides":
        output = output.reshape(2, 2).transpose(1, 0)

    with pytest.raises(ValueError):
        dist.all_to_all_single(output, input_tensor)


class ShapeMetadataTensor(Tensor):
    """Expose a supplied Python shape without allocating storage for that shape."""
    __fallback__ = None

    def __init__(self, metadata_shape):
        self.metadata_shape = metadata_shape
        super().__init__(np.zeros(1, dtype=np.float32))

    @property
    def shape(self):
        return self.metadata_shape


@pytest.mark.usefixtures("rank")
@pytest.mark.parametrize("negative_argument,output_splits,input_splits", [
    pytest.param("input", [1, 1], None, id="negative-input-omitted-input-splits"),
    pytest.param("output", None, [1, 1], id="negative-output-omitted-output-splits"),
])
@pytest.mark.parametrize("metadata_shape", [
    pytest.param((-2, 3), id="negative-first-dimension"),
    pytest.param((2, -3, 4), id="negative-middle-dimension"),
    pytest.param((2, 3, -4), id="negative-last-dimension"),
    pytest.param((0, -3), id="negative-after-zero-first-dimension"),
    pytest.param((2, 0, -3), id="negative-after-zero-middle-dimension"),
    pytest.param((2, -3, 0), id="negative-before-zero-dimension"),
])
def test_negative_fallback_dimensions_are_rejected_before_building_counts(
        negative_argument, output_splits, input_splits, metadata_shape):
    """
    Feature: mint.distributed.all_to_all_single Python metadata validation.
    Description: Report a negative dimension with omitted splits and explicitly split the other tensor.
    Expectation: Every negative dimension raises the same metadata error, including dimensions next to zero.
    """
    tensors = {
        "input": Tensor(np.zeros((2, 3), dtype=np.float32)),
        "output": Tensor(np.zeros((2, 3), dtype=np.float32)),
    }
    tensors[negative_argument] = ShapeMetadataTensor(metadata_shape)

    # Exercise preparation directly so that a regression never submits negative counts to HCCL.
    with pytest.raises(ValueError, match="input and output dimensions must be non-negative"):
        _validate_all_to_all_single_and_build_comm_args(
            tensors["output"], tensors["input"], output_splits, input_splits, GlobalComm.WORLD_COMM_GROUP, False)


@pytest.mark.usefixtures("rank")
@pytest.mark.parametrize("metadata_shape,splits", [
    pytest.param((2, 2**62), None, id="total-element-count-overflow"),
    pytest.param((2, 2**62, 4), [1, 1], id="row-size-overflow"),
    pytest.param((2, 2**62), [2, 0], id="peer-count-overflow"),
])
def test_nonnegative_fallback_dimensions_still_reject_count_overflow(metadata_shape, splits):
    """
    Feature: mint.distributed.all_to_all_single element count overflow protection.
    Description: Supply nonnegative dimensions that overflow the total, row or peer element count.
    Expectation: Count arithmetic raises ValueError after metadata validation succeeds.
    """
    input_tensor = ShapeMetadataTensor(metadata_shape)
    output = ShapeMetadataTensor(metadata_shape)

    with pytest.raises(ValueError, match="tensor element count is out of range"):
        _validate_all_to_all_single_and_build_comm_args(
            output, input_tensor, splits, splits, GlobalComm.WORLD_COMM_GROUP, False)


@pytest.mark.usefixtures("rank")
def test_fallback_counts_accept_int64_boundary_and_zero():
    """
    Feature: mint.distributed.all_to_all_single element count boundaries.
    Description: Prepare one row per peer using the largest int64 row size and a zero-sized output row.
    Expectation: Multiplication by one or zero preserves the exact boundary and produces zero receive counts.
    """
    max_count = np.iinfo(np.int64).max
    input_tensor = ShapeMetadataTensor((2, max_count))
    output = ShapeMetadataTensor((2, 0))

    args = _validate_all_to_all_single_and_build_comm_args(
        output, input_tensor, [1, 1], [1, 1], GlobalComm.WORLD_COMM_GROUP, False)

    assert args[3] == (max_count, max_count)
    assert args[4] == (0, 0)


def test_rejected_split_does_not_poison_the_next_collective(rank):
    """
    Feature: mint.distributed.all_to_all_single validation recovery.
    Description: Reject an incorrect split length, then perform a valid exchange in the same process.
    Expectation: The first call raises ValueError and the next collective completes with the expected values.
    """
    input_tensor = Tensor(SEND_VALUES[rank], ms.float32)
    output = Tensor([-1, -1, -1, -1], ms.float32)
    with pytest.raises(ValueError):
        dist.all_to_all_single(output, input_tensor, input_split_sizes=[4])

    dist.all_to_all_single(output, input_tensor)

    np.testing.assert_array_equal(output.asnumpy(), RECEIVED_VALUES[rank])


@pytest.mark.parametrize("dtype, send_values, received_values", DISPATCH_CASES)
def test_tensor_fallback_still_intercepts_the_communication_operator(rank, dtype, send_values, received_values):
    """
    Feature: mint.distributed.all_to_all_single Tensor fallback.
    Description: Exchange real or complex vectors and matrices with Tensor subclass fallback.
    Expectation: Fallback intercepts the views and communication operator while preserving output values and dtype.
    """
    from tests.st.pynative.framework.test_fallback import DTensor, Layout

    input_tensor = DTensor(Tensor(send_values[rank], dtype), Layout())
    output = DTensor(Tensor(np.full(np.shape(received_values[rank]), -1), dtype), Layout())
    DTensor.RUN_OPS.clear()

    dist.all_to_all_single(output, input_tensor)

    assert DTensor.RUN_OPS.count("DistCommAllToAllVSingle") == 1
    if dtype in (ms.complex64, ms.complex128):
        assert DTensor.RUN_OPS[:2] == ["ViewDtype", "ViewDtype"]
    assert output.dtype == dtype
    assert output.shape == np.shape(received_values[rank])
    np.testing.assert_array_equal(output.asnumpy(), received_values[rank])


@pytest.mark.parametrize("dtype, send_values, received_values", DISPATCH_CASES)
def test_tensor_dispatch_still_intercepts_the_communication_operator(rank, dtype, send_values, received_values):
    """
    Feature: mint.distributed.all_to_all_single Tensor dispatch.
    Description: Exchange real or complex vectors and matrices with Tensor subclass dispatch.
    Expectation: Dispatch intercepts the views and communication operator while preserving output values and dtype.
    """
    from tests.st.pynative.framework.test_ms_dispatch import DTensor, Layout, RUN_OPS

    input_tensor = DTensor(Tensor(send_values[rank], dtype).to("Ascend"), Layout())
    output = DTensor(Tensor(np.full(np.shape(received_values[rank]), -1), dtype).to("Ascend"), Layout())
    RUN_OPS.clear()

    dist.all_to_all_single(output, input_tensor)

    assert RUN_OPS.count("DistCommAllToAllVSingle") == 1
    if dtype in (ms.complex64, ms.complex128):
        assert RUN_OPS[:2] == ["ViewDtype", "ViewDtype"]
    assert output.dtype == dtype
    assert output.shape == np.shape(received_values[rank])
    np.testing.assert_array_equal(output.asnumpy(), received_values[rank])


@pytest.mark.parametrize("dtype, send_values, received_values", DISPATCH_CASES)
def test_dispatch_mode_intercepts_communication_on_regular_tensors(rank, dtype, send_values, received_values):
    """
    Feature: mint.distributed.all_to_all_single dispatch mode.
    Description: Exchange ordinary real or complex vectors and matrices while a dispatch mode records operators.
    Expectation: The active mode observes communication and the existing output receives the expected values.
    """
    from tests.st.pynative.framework.test_ms_dispatch import OpTracer

    input_tensor = Tensor(send_values[rank], dtype).to("Ascend")
    output = Tensor(np.full(np.shape(received_values[rank]), -1), dtype).to("Ascend")
    tracer = OpTracer()

    with tracer:
        result = dist.all_to_all_single(output, input_tensor)

    assert result is None
    assert tracer.fwd_ops.count("DistCommAllToAllVSingle") == 1
    if dtype in (ms.complex64, ms.complex128):
        assert tracer.fwd_ops[:2] == ["ViewDtype", "ViewDtype"]
    assert output.dtype == dtype
    assert output.shape == np.shape(received_values[rank])
    np.testing.assert_array_equal(output.asnumpy(), received_values[rank])

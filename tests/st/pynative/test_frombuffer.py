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
""" test_frombuffer """
import gc
import weakref
import sys
import numpy as np
import mindspore as ms
from mindspore import Tensor
from tests.mark_utils import arg_mark

# Tests for the `frombuffer` function (only work on CPU):
# Constructs tensors from Python objects that implement the buffer protocol,
# without copying data.
SIZE = 5
SHAPE = (SIZE,)

numpy_to_mindspore_dtype_dict = {
    np.bool_: ms.bool,
    np.int8: ms.int8,
    np.int16: ms.int16,
    np.int32: ms.int32,
    np.int64: ms.int64,
    np.uint8: ms.uint8,
    np.uint16: ms.uint16,
    np.uint32: ms.uint32,
    np.uint64: ms.uint64,
    np.float16: ms.float16,
    np.float32: ms.float32,
    np.float64: ms.float64,
    np.complex64: ms.complex64,
    np.complex128: ms.complex128
}

mindspore_to_numpy_dtype_dict = {v: k for k, v in numpy_to_mindspore_dtype_dict.items()}

ALL_TEST_DTYPES = list(set(mindspore_to_numpy_dtype_dict.keys()))

def get_dtype_size(dtype):
    return int(Tensor([], dtype=dtype).asnumpy().dtype.itemsize)

def make_tensor(shape, dtype):
    np_dtype = mindspore_to_numpy_dtype_dict[dtype]

    if np_dtype == np.bool_:
        np_array = np.random.randint(0, 2, size=shape).astype(np.bool_)
    else:
        np_array = np.random.standard_normal(shape).astype(np_dtype)
    return Tensor(np_array, dtype=dtype)

def dtypes(*dtype_list):
    def decorator(test_func):
        def wrapper(self, *args, **kwargs):
            for dtype in dtype_list:
                try:
                    test_func(self, dtype, *args, **kwargs)
                except Exception as e:
                    raise AssertionError(f"Test failed for dtype: {dtype}") from e
        return wrapper
    return decorator

@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
class TestBufferProtocol:
    """
    Test buffer protocol support in frombuffer function
    """
    def _run_test(self, shape, dtype, count=-1, offset=0):
        """
        Run a single test case for frombuffer
        """
        numpy_dtype = mindspore_to_numpy_dtype_dict[dtype]

        numpy_original = make_tensor(shape, dtype=dtype).asnumpy()
        original = memoryview(numpy_original)

        ms_frombuffer = ms.frombuffer(original, dtype=dtype, count=count, offset=offset)
        np_frombuffer = np.frombuffer(original, dtype=numpy_dtype, count=count, offset=offset)

        ms_array = ms_frombuffer.asnumpy()
        np.testing.assert_array_equal(np_frombuffer, ms_array)

        if np_frombuffer.size > 0:
            test_value = np.array([123], dtype=numpy_dtype)[0] # use scalar value to ensure compatibility
            ms_array[0] = test_value
            expected_index = offset // get_dtype_size(dtype)
            assert numpy_original.reshape(-1)[expected_index] == test_value, "Data not shared between buffer and tensor"

        return (numpy_original, ms_frombuffer)

    @dtypes(*ALL_TEST_DTYPES)
    def test_same_type(self, dtype):
        """
        Test frombuffer with same type as buffer
        """
        self._run_test((), dtype)
        self._run_test((4,), dtype)
        self._run_test((10, 10), dtype)

    @dtypes(*ALL_TEST_DTYPES)
    def test_with_offset(self, dtype):
        """
        Test frombuffer with offset
        """
        dtype_size = get_dtype_size(dtype)
        for i in range(SIZE):
            self._run_test(SHAPE, dtype, offset=i * dtype_size)

    @dtypes(*ALL_TEST_DTYPES)
    def test_with_count(self, dtype):
        """
        Test frombuffer with count
        """
        for i in range(-1, SIZE + 1):
            if i != 0:
                self._run_test(SHAPE, dtype, count=i)

    @dtypes(*ALL_TEST_DTYPES)
    def test_with_count_and_offset(self, dtype):
        """
        Test frombuffer with count and offset
        """
        dtype_size = get_dtype_size(dtype)
        for count in range(1, SIZE):
            for offset in range(0, (SIZE - count) * dtype_size, dtype_size):
                self._run_test(SHAPE, dtype, count=count, offset=offset)

    @dtypes(*ALL_TEST_DTYPES)
    def test_invalid_arguments(self, dtype):
        """
        Test frombuffer with invalid arguments
        """
        dtype_size = get_dtype_size(dtype)
        total_bytes = SIZE * dtype_size
        total_elements = SIZE

        test_cases = [
            ("Empty array", lambda: ms.frombuffer(memoryview(np.array([], dtype=mindspore_to_numpy_dtype_dict[dtype])), dtype=dtype)),
            ("Count equals 0", lambda: self._run_test(SHAPE, dtype, count=0)),
            ("Offset negative", lambda: self._run_test(SHAPE, dtype, offset=-dtype_size)),
            ("Offset bigger than total length", lambda: self._run_test(SHAPE, dtype, offset=total_bytes)),
            ("Count negative other than -1", lambda: self._run_test(SHAPE, dtype, count=-2)),
            ("Count too large for buffer", lambda: self._run_test(SHAPE, dtype, count=total_elements + 1)),
            ("Offset leaves insufficient space", lambda: self._run_test(SHAPE, dtype, offset=total_bytes - 1, count=2)),
        ]

        if dtype_size > 1:
            test_cases.append(("Non-multiple offset with all elements", lambda: self._run_test(SHAPE, dtype, offset=dtype_size - 1)))

        for case_name, test_func in test_cases:
            try:
                test_func()
                assert False, f"Test case '{case_name}' should have raised an exception"
            except RuntimeError:
                pass

    @dtypes(*ALL_TEST_DTYPES)
    def test_shared_buffer(self, dtype):
        """
        Test that the tensor shares memory with the original buffer
        """
        numpy_dtype = mindspore_to_numpy_dtype_dict[dtype]
        test_val = np.array([123], dtype=numpy_dtype)[0]

        arr, tensor = self._run_test(SHAPE, dtype)
        tensor_array = tensor.asnumpy()
        tensor_array[:] = test_val
        np.testing.assert_array_equal(arr.reshape(-1), tensor_array)

        for count in [-1, 1, 2, 3]:
            if count == 0:
                continue
            actual_count = count if count > 0 else SIZE
            dtype_size = get_dtype_size(dtype)
            max_offset = (SIZE - actual_count) * dtype_size

            for offset in range(0, max_offset + 1, dtype_size):
                arr, tensor = self._run_test(SHAPE, dtype, count=count, offset=offset)
                tensor_array = tensor.asnumpy()
                tensor_array[:] = test_val

                start_idx = offset // dtype_size
                end_idx = start_idx + actual_count
                np.testing.assert_array_equal(arr.reshape(-1)[start_idx:end_idx], tensor_array)

    def test_not_a_buffer(self):
        """
        Test frombuffer with non-buffer objects
        """
        for dtype in ALL_TEST_DTYPES:
            try:
                ms.frombuffer([1, 2, 3, 4], dtype=dtype)
                assert False, "Should have raised an exception for non-buffer object"
            except RuntimeError:
                pass

    def test_non_writable_buffer(self):
        """
        Test frombuffer with non-writable buffer
        """
        for dtype in ALL_TEST_DTYPES:
            numpy_arr = make_tensor((1,), dtype=dtype).asnumpy()
            byte_arr = numpy_arr.tobytes()
            tensor = ms.frombuffer(byte_arr, dtype=dtype)
            assert tensor is not None

    def test_byte_to_int(self):
        """
        Test frombuffer converting byte buffer to int tensor
        """
        if sys.byteorder == 'little':
            byte_array = np.array([-1, 0, 0, 0, -1, 0, 0, 0], dtype=np.byte)
        else:
            byte_array = np.array([0, 0, 0, -1, 0, 0, 0, -1], dtype=np.byte)

        tensor = ms.frombuffer(byte_array, dtype=ms.int32)
        tensor_array = tensor.asnumpy()

        assert tensor_array.size == 2
        assert (tensor_array == 255).all()

    def test_destruction(self):
        """
        Test that the buffer is not destroyed while the tensor exists
        """
        class TrackedBuffer(bytearray):
            destroyed = False
            def __del__(self):
                TrackedBuffer.destroyed = True

        buffer = TrackedBuffer(b'\x01\x02\x03\x04\x05')
        buffer_ref = weakref.ref(buffer)
        tensor = ms.frombuffer(buffer, dtype=ms.uint8)

        del buffer
        gc.collect()
        assert not TrackedBuffer.destroyed and buffer_ref() is not None, "buffer should not be destroyed yet"

        del tensor
        gc.collect()
        assert TrackedBuffer.destroyed or buffer_ref() is None, "buffer should be destroyed after tensor deletion"

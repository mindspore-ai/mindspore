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
# ==============================================================================
"""Test collate function."""

import collections
import copy
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset.dataloader import default_collate, default_convert
from mindspore.dataset.dataloader._utils.collate import collate, default_collate_fn_map
from tests.mark_utils import arg_mark


class MutableMappingWithExtraProperty(collections.abc.MutableMapping):
    """A mutable mapping with some extra properties."""

    def __init__(self, data: collections.abc.Mapping, extra_property: Any = None):
        self._data = data
        self.extra_property = extra_property

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __copy__(self):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        new_obj._data = copy.copy(self._data)
        new_obj.extra_property = copy.copy(self.extra_property)
        return new_obj


class MutableSequenceWithExtraProperty(collections.abc.MutableSequence):
    """A mutable sequence with some extra properties."""

    def __init__(self, data: collections.abc.Sequence, extra_property: Any = None):
        self._data = data
        self.extra_property = extra_property

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __delitem__(self, index):
        del self._data[index]

    def insert(self, index, value):
        self._data.insert(index, value)

    def __len__(self):
        return len(self._data)

    def __copy__(self):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        new_obj._data = copy.copy(self._data)
        new_obj.extra_property = copy.copy(self.extra_property)
        return new_obj


class TestDefaultConvert:
    """Test default_convert function."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            ms.Tensor(True, dtype=ms.bool),
            ms.Tensor([0], dtype=ms.uint8),
            ms.Tensor([[-1], [1]], dtype=ms.int32),
            ms.Tensor([[[3.14], [-3.14]], [[3.14], [-3.14]]], dtype=ms.float32),
            ms.Tensor([1 + 2j], dtype=ms.complex64),
        ),
    )
    def test_convert_tensor(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with Tensor.
        Expectation: The result is unchanged.
        """
        result = default_convert(data)
        assert isinstance(result, ms.Tensor)
        np.testing.assert_equal(data.asnumpy(), result.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            np.bool_(True),
            np.uint8(0),
            np.int32(-1),
            np.float32(3.14),
            np.complex64(1 + 2j),
        ),
    )
    def test_convert_numpy_scalar(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with numpy primitive type that supports conversion.
        Expectation: The result is a Tensor.
        """
        result = default_convert(data)
        assert isinstance(result, ms.Tensor)
        np.testing.assert_equal(data, result.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize("data", (np.str_("abc"), np.bytes_(b"0xffff")))
    def test_convert_unsupported_numpy_scalar(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with numpy primitive type that does not support conversion.
        Expectation: The result is unchanged.
        """
        result = default_convert(data)
        assert isinstance(result, type(data))
        assert result == data

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            np.array([True], dtype=np.bool_),
            np.array([0], dtype=np.uint8),
            np.array([[-1], [1]], dtype=np.int32),
            np.array([[[3.14], [-3.14]], [[3.14], [-3.14]]], dtype=np.float32),
            np.array([1 + 2j], dtype=np.complex64),
        ),
    )
    def test_convert_numpy_array(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with numpy array that supports conversion.
        Expectation: The result is a Tensor.
        """
        result = default_convert(data)
        assert isinstance(result, ms.Tensor)
        np.testing.assert_equal(data, result.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            np.array([b"abc"], dtype=np.bytes_),
            np.array(["abc"], dtype=np.str_),
            np.array([{"data": "abc"}], dtype=np.object_),
        ),
    )
    def test_convert_unsupported_numpy_array(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with numpy array that does not support conversion.
        Expectation: The result is unchanged.
        """
        result = default_convert(data)
        assert isinstance(result, type(data))
        np.testing.assert_equal(data, result)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            MutableMappingWithExtraProperty(
                {"uint8": np.uint8(0), "int32": np.int32(-1), "float32": np.float32(3.14)}, extra_property="converted"
            ),
            MutableMappingWithExtraProperty({"str": "a", "bytes": b"b"}, extra_property="unchanged"),
        ),
    )
    def test_convert_mutable_mapping_with_extra_property(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with mutable mapping that has extra property.
        Expectation: The result is still the original type and the extra property is preserved.
        """
        result = default_convert(data)
        assert isinstance(result, MutableMappingWithExtraProperty)
        assert result.extra_property == data.extra_property
        assert result.keys() == data.keys()
        for key, value in result.items():
            expected_value = default_convert(data[key])
            assert isinstance(value, type(expected_value))
            if isinstance(value, ms.Tensor):
                np.testing.assert_equal(value.asnumpy(), expected_value.asnumpy())
            else:
                assert value == expected_value

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            {"1": np.int32(1), "2": np.int32(2), "3": np.int32(3)},
            collections.OrderedDict({"1": np.int32(1), "2": np.int32(2), "3": np.int32(3)}),
            collections.defaultdict(np.int32, {"1": np.int32(1), "2": np.int32(2), "3": np.int32(3)}),
            MappingProxyType({"1": np.int32(1), "2": np.int32(2), "3": np.int32(3)}),
        ),
    )
    def test_convert_mapping(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with mapping.
        Expectation: The result is still the original type but the values have been converted to Tensor.
        """
        result = default_convert(data)
        assert isinstance(result, type(data))
        assert result.keys() == data.keys()
        for key, value in result.items():
            assert isinstance(value, ms.Tensor)
            np.testing.assert_equal(data[key], value.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_convert_mapping_not_support_copy(self, monkeypatch):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with mapping that not support copy.
        Expectation: The result is a dict with keys unchanged and values converted to Tensor.
        """

        def unimplemented_copy(self):
            raise TypeError

        monkeypatch.setattr(MutableMappingWithExtraProperty, "__copy__", unimplemented_copy)
        data = MutableMappingWithExtraProperty({"uint8": np.uint8(0)}, extra_property="converted")
        result = default_convert(data)
        assert isinstance(result, dict)
        assert result.keys() == data.keys()
        for key, value in result.items():
            assert isinstance(value, ms.Tensor)
            np.testing.assert_equal(data[key], value.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            collections.namedtuple("User", ["name", "age"])("John", 20),
            collections.namedtuple("Media", ["image", "audio", "label"])(
                ms.Tensor([[0, 1], [2, 3]]), np.array([[4, 5], [6, 7]]), np.uint8(0)
            ),
        ),
    )
    def test_convert_namedtuple(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with namedtuple.
        Expectation: The result is still the original type but the values may be converted.
        """
        result = default_convert(data)
        assert isinstance(result, type(data))
        assert result._fields == data._fields
        for field in result._fields:
            data_value = getattr(data, field)
            expected_value = default_convert(data_value)
            result_value = getattr(result, field)
            assert isinstance(result_value, type(expected_value))
            if isinstance(result_value, ms.Tensor):
                np.testing.assert_equal(result_value.asnumpy(), expected_value.asnumpy())
            else:
                assert result_value == expected_value

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            (1, 0.5, False, "word", b"bytes"),
            (np.float32(1.5), np.array([1, 1, 1]), ms.Tensor([[[0]]])),
        ),
    )
    def test_convert_tuple(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with tuple.
        Expectation: The result is a list but the values may be converted.
        """
        result = default_convert(data)
        assert isinstance(result, list)
        assert len(result) == len(data)
        for result_value, data_value in zip(result, data):
            expected_value = default_convert(data_value)
            assert isinstance(result_value, type(expected_value))
            if isinstance(result_value, ms.Tensor):
                np.testing.assert_equal(result_value.asnumpy(), expected_value.asnumpy())
            else:
                assert result_value == expected_value

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            MutableSequenceWithExtraProperty([np.uint8(0), np.int32(-1), np.float32(3.14)], extra_property="converted"),
            MutableSequenceWithExtraProperty(["a", b"b"], extra_property="unchanged"),
        ),
    )
    def test_convert_mutable_sequence_with_extra_property(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with mutable sequence that has extra property.
        Expectation: The result is still the original type and the extra property is preserved.
        """
        result = default_convert(data)
        assert isinstance(result, MutableSequenceWithExtraProperty)
        assert result.extra_property == data.extra_property
        assert len(result) == len(data)
        for result_value, data_value in zip(result, data):
            expected_value = default_convert(data_value)
            assert isinstance(result_value, type(expected_value))
            if isinstance(result_value, ms.Tensor):
                np.testing.assert_equal(result_value.asnumpy(), expected_value.asnumpy())
            else:
                assert result_value == expected_value

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            [np.int32(1), np.array([1, 2, 3]), ms.Tensor([[0, 1], [2, 3]])],
            bytearray("123", encoding="utf-8"),
        ),
    )
    def test_convert_sequence(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with sequence.
        Expectation: The result is still the original type but the values may be converted.
        """
        result = default_convert(data)
        assert isinstance(result, type(data))
        assert len(data) == len(result)
        for result_value, data_value in zip(result, data):
            expected_value = default_convert(data_value)
            assert isinstance(result_value, type(expected_value))
            if isinstance(result_value, ms.Tensor):
                np.testing.assert_equal(result_value.asnumpy(), expected_value.asnumpy())
            else:
                assert result_value == expected_value

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            range(0, 10, 2),
            memoryview(b"\xb9\x01\xef"),
        ),
    )
    def test_convert_sequence_not_support_reconstruct(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with sequence that not support directly reconstruct.
        Expectation: The result is a list but the values may be converted.
        """
        result = default_convert(data)
        assert isinstance(result, list)
        assert len(data) == len(result)
        for result_value, data_value in zip(result, data):
            expected_value = default_convert(data_value)
            assert isinstance(result_value, type(expected_value))
            if isinstance(result_value, ms.Tensor):
                np.testing.assert_equal(result_value.asnumpy(), expected_value.asnumpy())
            else:
                assert result_value == expected_value

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize("data", (0, -1.5, True, 4 + 3j, "abc", b"abc", {0, 1, 2}, object()))
    def test_convert_unsupported_type(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with unsupported type.
        Expectation: The result is unchanged.
        """
        result = default_convert(data)
        assert isinstance(result, type(data))
        assert result == data


class TestDefaultCollate:
    """Test default_collate function."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            # same dtype
            [ms.Tensor(True, dtype=ms.bool), ms.Tensor(False, dtype=ms.bool)],
            [ms.Tensor([0], dtype=ms.uint8), ms.Tensor([1], dtype=ms.uint8)],
            [ms.Tensor([[-1], [1]], dtype=ms.int32), ms.Tensor([[-2], [2]], dtype=ms.int32)],
            [ms.Tensor([[[3.14], [-3.14]]], dtype=ms.float32), ms.Tensor([[[-3.14], [3.14]]], dtype=ms.float32)],
            [ms.Tensor([1 + 2j], dtype=ms.complex64), ms.Tensor([1 - 2j], dtype=ms.complex64)],
            # different dtype
            [ms.Tensor(True, dtype=ms.bool), ms.Tensor(1, dtype=ms.uint8), ms.Tensor(-1, dtype=ms.int32)],
            [ms.Tensor([3.14], dtype=ms.float16), ms.Tensor([1.414], dtype=ms.float32)],
            [ms.Tensor([1 + 2j], dtype=ms.complex64), ms.Tensor([0.5 - 0.5j], dtype=ms.complex128)],
        ),
    )
    def test_collate_tensor(self, data):
        """
        Feature: Test default_collate function.
        Description: Test default_collate function with Tensor.
        Expectation: The result is a Tensor with an extra dimension that concatenates the inputs.
        """
        result = default_collate(data)
        assert isinstance(result, ms.Tensor)
        np.testing.assert_equal(ms.Tensor(data).asnumpy(), result.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            # same dtype
            [np.array(True, dtype=np.bool_), np.array(False, dtype=np.bool_)],
            [np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8)],
            [np.array([[-1], [1]], dtype=np.int32), np.array([[-2], [2]], dtype=np.int32)],
            [np.array([[[3.14], [-3.14]]], dtype=np.float32), np.array([[[-3.14], [3.14]]], dtype=np.float32)],
            [np.array([1 + 2j], dtype=np.complex64), np.array([1 - 2j], dtype=np.complex64)],
            # different dtype
            [np.array(True, dtype=np.bool_), np.array(1, dtype=np.uint8), np.array(-1, dtype=np.int32)],
            [np.array([3.14], dtype=np.float16), np.array([1.414], dtype=np.float32)],
            [np.array([1 + 2j], dtype=np.complex64), np.array([0.5 - 0.5j], dtype=np.complex128)],
        ),
    )
    def test_collate_numpy_array(self, data):
        """
        Feature: Test default_collate function.
        Description: Test default_collate function with numpy array.
        Expectation: The result is a Tensor with an extra dimension that concatenates the inputs.
        """
        result = default_collate(data)
        assert isinstance(result, ms.Tensor)
        np.testing.assert_equal(ms.Tensor(data).asnumpy(), result.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            [np.array([b"abc"], dtype=np.bytes_)],
            [np.array(["abc"], dtype=np.str_)],
            [np.array([{"data": "abc"}], dtype=np.object_)],
        ),
    )
    def test_collate_unsupported_numpy_array(self, data):
        """
        Feature: Test default_collate function.
        Description: Test default_collate function with unsupported numpy array.
        Expectation: Raise TypeError.
        """
        with pytest.raises(
            TypeError,
            match=f"NumPy arrays with dtype {data[0].dtype} are not supported for collation",
        ):
            default_collate(data)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            # same dtype
            [np.bool_(True), np.bool_(False)],
            [np.uint8(0), np.uint8(1)],
            [np.int32(-1), np.int32(-2)],
            [np.float32(3.14), np.float32(1.414)],
            [np.complex64(1 + 2j), np.complex64(1 - 2j)],
            # different dtype
            [np.bool_(True), np.int16(-5), np.uint64(128)],
            [np.float16(0.1), np.float64(-3.5)],
            [np.complex64(1j), np.complex128(-2j)],
        ),
    )
    def test_collate_numpy_scalar(self, data):
        """
        Feature: Test default_collate function.
        Description: Test default_collate function with numpy scalar.
        Expectation: The result is a Tensor with an extra dimension that concatenates the inputs.
        """
        result = default_collate(data)
        assert isinstance(result, ms.Tensor)
        np.testing.assert_equal(ms.Tensor(data).asnumpy(), result.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            [0.1, 0.2, 0.3],
            [32.3 + 1e1, 32.3 - 1e1],
        ),
    )
    def test_collate_float(self, data):
        """
        Feature: Test default_collate function.
        Description: Test default_collate function with float.
        Expectation: The result is a Tensor of dtype float64 with an extra dimension that concatenates the inputs.
        """
        result = default_collate(data)
        assert isinstance(result, ms.Tensor)
        assert result.dtype == ms.float64
        np.testing.assert_allclose(ms.Tensor(data).asnumpy(), result.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize("data", ([1, -2, 3],))
    def test_collate_int(self, data):
        """
        Feature: Test default_collate function.
        Description: Test default_collate function with int.
        Expectation: The result is a Tensor with an extra dimension that concatenates the inputs.
        """
        result = default_collate(data)
        assert isinstance(result, ms.Tensor)
        np.testing.assert_equal(ms.Tensor(data).asnumpy(), result.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            ["a", "b", "c"],
            [b"a", b"b", b"c"],
            ["1", b"2"],
        ),
    )
    def test_collate_str(self, data):
        """
        Feature: Test default_collate function.
        Description: Test default_collate function with str and bytes.
        Expectation: The result is unchanged.
        """
        result = default_collate(data)
        assert isinstance(result, type(data))
        assert result == data


class TestCollate:
    """Test collate function."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize("data", ([1, 2, 3],))
    def test_collate_without_collate_fn_map(self, data):
        """
        Feature: Test collate function.
        Description: Test collate function without collate_fn_map.
        Expectation: Raise TypeError.
        """
        with pytest.raises(TypeError, match=f"Cannot find the appropriate collate function for type {type(data[0])}"):
            collate(data, collate_fn_map=None)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            [
                MutableMappingWithExtraProperty(
                    {"uint8": np.uint8(0), "int32": np.int32(-1), "float32": np.float32(3.141)},
                    extra_property="converted",
                ),
                MutableMappingWithExtraProperty(
                    {"uint8": np.uint8(1), "int32": np.int32(-2), "float32": np.float32(1.414)},
                    extra_property="converted",
                ),
            ],
        ),
    )
    def test_collate_mutable_mapping_with_extra_property(self, data):
        """
        Feature: Test collate function.
        Description: Test collate function with mutable mapping that has extra property.
        Expectation: The result is still the original type and the extra property is preserved.
        """
        result = collate(data, collate_fn_map=default_collate_fn_map)
        assert isinstance(result, MutableMappingWithExtraProperty)
        assert result.extra_property == data[0].extra_property
        assert result.keys() == data[0].keys()
        for key in result.keys():
            assert isinstance(result[key], ms.Tensor)
            np.testing.assert_equal(ms.Tensor([d[key] for d in data]).asnumpy(), result[key].asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            [
                {"int32": np.int32(1), "float32": np.float32(0.1)},
                {"int32": np.int32(2), "float32": np.float32(0.2)},
            ],
            [
                collections.OrderedDict({"int32": np.int32(1), "float32": np.float32(0.1)}),
                collections.OrderedDict({"int32": np.int32(2), "float32": np.float32(0.2)}),
            ],
            [
                collections.defaultdict(None, {"int32": np.int32(1), "float32": np.float32(0.1)}),
                collections.defaultdict(None, {"int32": np.int32(2), "float32": np.float32(0.2)}),
            ],
            [
                MappingProxyType({"int32": np.int32(1), "float32": np.float32(0.1)}),
                MappingProxyType({"int32": np.int32(2), "float32": np.float32(0.2)}),
            ],
        ),
    )
    def test_collate_mapping(self, data):
        """
        Feature: Test collate function.
        Description: Test collate function with mapping.
        Expectation: The result is still the original type but the values have been concatenated.
        """
        result = collate(data, collate_fn_map=default_collate_fn_map)
        assert isinstance(result, type(data[0]))
        assert result.keys() == data[0].keys()
        for key in result.keys():
            assert isinstance(result[key], ms.Tensor)
            np.testing.assert_equal(ms.Tensor([d[key] for d in data]).asnumpy(), result[key].asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_collate_mapping_not_support_copy(self, monkeypatch):
        """
        Feature: Test collate function.
        Description: Test collate function with mapping that not support copy.
        Expectation: The result is a dict with keys unchanged and values have been concatenated.
        """

        def unimplemented_copy(self):
            raise TypeError

        monkeypatch.setattr(MutableMappingWithExtraProperty, "__copy__", unimplemented_copy)
        data = [
            MutableMappingWithExtraProperty({"uint8": np.uint8(0)}, extra_property="converted"),
            MutableMappingWithExtraProperty({"uint8": np.uint8(1)}, extra_property="converted"),
        ]
        result = collate(data, collate_fn_map=default_collate_fn_map)
        assert isinstance(result, dict)
        assert result.keys() == data[0].keys()
        for key, value in result.items():
            assert isinstance(value, ms.Tensor)
            np.testing.assert_equal(ms.Tensor([d[key] for d in data]).asnumpy(), value.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            [
                collections.namedtuple("Media", ["image", "audio", "label"])(
                    ms.Tensor([[0, 1], [2, 3]]), np.array([[4, 5], [6, 7]]), np.uint8(0)
                ),
                collections.namedtuple("Media", ["image", "audio", "label"])(
                    ms.Tensor([[0, -1], [-2, -3]]), np.array([[-4, -5], [-6, -7]]), np.uint8(1)
                ),
            ],
        ),
    )
    def test_collate_namedtuple(self, data):
        """
        Feature: Test collate function.
        Description: Test collate function with namedtuple.
        Expectation: The result is still the original type but the values have been concatenated.
        """
        result = collate(data, collate_fn_map=default_collate_fn_map)
        assert isinstance(result, type(data[0]))
        assert result._fields == data[0]._fields
        for field in result._fields:
            data_value = [getattr(d, field) for d in data]
            expected_value = ms.Tensor(data_value)
            result_value = getattr(result, field)
            assert isinstance(result_value, ms.Tensor)
            np.testing.assert_equal(result_value.asnumpy(), expected_value.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            [
                (1, 0.5, False, "word", b"bytes"),
                (-1, -0.5, True, "WORD", b"BYTES"),
            ],
        ),
    )
    def test_collate_tuple(self, data):
        """
        Feature: Test collate function.
        Description: Test collate function with tuple.
        Expectation: The result is a list but the values have been concatenated.
        """
        result = collate(data, collate_fn_map=default_collate_fn_map)
        assert isinstance(result, list)
        assert len(result) == len(data[0])
        for index, value in enumerate(result):
            expected_value = collate(tuple(d[index] for d in data), collate_fn_map=default_collate_fn_map)
            assert isinstance(value, type(expected_value))
            if isinstance(value, ms.Tensor):
                np.testing.assert_equal(value.asnumpy(), expected_value.asnumpy())
            else:
                assert value == expected_value

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize("data", ([(0, 1, 2), (3, 4, 5, 6)],))
    def test_collate_tuple_with_different_length(self, data):
        """
        Feature: Test collate function.
        Description: Test collate function with tuple that has different length.
        Expectation: Raise RuntimeError.
        """
        with pytest.raises(RuntimeError, match="Each element in list of batch must be of equal size"):
            collate(data, collate_fn_map=default_collate_fn_map)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            [
                MutableSequenceWithExtraProperty(
                    [np.uint8(0), np.int32(-1), np.float32(3.14)], extra_property="converted"
                ),
                MutableSequenceWithExtraProperty(
                    [np.uint8(1), np.int32(-2), np.float32(1.414)], extra_property="converted"
                ),
            ],
        ),
    )
    def test_collate_mutable_sequence_with_extra_property(self, data):
        """
        Feature: Test collate function.
        Description: Test collate function with mutable sequence that has extra property.
        Expectation: The result is still the original type and the extra property is preserved.
        """
        result = collate(data, collate_fn_map=default_collate_fn_map)
        assert isinstance(result, MutableSequenceWithExtraProperty)
        assert result.extra_property == data[0].extra_property
        for index, value in enumerate(result):
            expected_value = collate(tuple(d[index] for d in data), collate_fn_map=default_collate_fn_map)
            assert isinstance(value, type(expected_value))
            np.testing.assert_equal(value.asnumpy(), expected_value.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            [
                [np.int32(1), np.array([1, 2, 3]), ms.Tensor([[0, 1], [2, 3]])],
                [np.int32(2), np.array([4, 5, 6]), ms.Tensor([[3, 4], [5, 6]])],
            ],
            [
                bytearray(),
                bytearray(),
            ],
        ),
    )
    def test_collate_sequence(self, data):
        """
        Feature: Test collate function.
        Description: Test collate function with sequence.
        Expectation: The result is still the original type but the values have been concatenated.
        """
        result = collate(data, collate_fn_map=default_collate_fn_map)
        assert isinstance(result, type(data[0]))
        assert len(result) == len(data[0])
        for index, value in enumerate(result):
            expected_value = collate(tuple(d[index] for d in data), collate_fn_map=default_collate_fn_map)
            assert isinstance(value, type(expected_value))
            np.testing.assert_equal(value.asnumpy(), expected_value.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            [range(0, 10, 2), range(10, 20, 2)],
            [memoryview(b"\xb9\x01\xef"), memoryview(b"\xef\x01\xb9")],
        ),
    )
    def test_collate_sequence_not_support_reconstruct(self, data):
        """
        Feature: Test collate function.
        Description: Test collate function with sequence that not support directly reconstruct.
        Expectation: The result is a list but the values have been concatenated.
        """
        result = collate(data, collate_fn_map=default_collate_fn_map)
        assert isinstance(result, list)
        assert len(result) == len(data[0])
        for index, value in enumerate(result):
            expected_value = collate(tuple(d[index] for d in data), collate_fn_map=default_collate_fn_map)
            assert isinstance(value, type(expected_value))
            np.testing.assert_equal(value.asnumpy(), expected_value.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            [4 + 3j, 4 - 3j],
            [object(), object()],
        ),
    )
    def test_collate_unsupported_type(self, data):
        """
        Feature: Test collate function.
        Description: Test collate function with unsupported type.
        Expectation: Raise TypeError.
        """
        with pytest.raises(
            TypeError,
            match=f"Cannot find the appropriate collate function for type {type(data[0])}",
        ):
            collate(data, collate_fn_map=default_collate_fn_map)

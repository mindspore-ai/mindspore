# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Stub Tensor implementation."""

from functools import reduce
from mindspore.common.tensor import Tensor
from mindspore.common.dtype import type_size_in_bytes
from mindspore._c_expression import Tensor as Tensor_


class StubTensor(Tensor):
    """stub tensor for async op run."""

    def __init__(self, stub):
        Tensor.__init__(self, internal=True)
        self.stub = stub

    def __repr__(self):
        self.stub_sync()
        return super().__repr__()

    def __str__(self):
        self.stub_sync()
        return super().__str__()

    def __setitem__(self, index, value):
        self.stub_sync()
        return super().__setitem__(index, value)

    @property
    def shape(self):
        """shape stub."""
        if self.stub:
            return self.stub.get_shape()
        return super().shape

    @property
    def dtype(self):
        """dtype stub."""
        if self.stub:
            return self.stub.get_dtype()
        return super().dtype

    @property
    def size(self):
        """size stub."""
        shape = self.shape
        return reduce((lambda x, y: x * y), shape) if shape else 1

    @property
    def itemsize(self):
        """itemsize stub."""
        return type_size_in_bytes(self.dtype)

    @property
    def nbytes(self):
        """nbytes stub."""
        return self.size * self.itemsize

    @property
    def ndim(self):
        """ndim stub."""
        return len(self.shape)

    @property
    def strides(self):
        """strides stub."""
        self.stub_sync()
        return super().strides

    @property
    def has_init(self):
        """has_init stub."""
        self.stub_sync()
        return super().has_init

    def stub_sync(self):
        """data sync to get real tensor"""
        if self.stub:
            val = self.stub.get_value()
            Tensor_.__init__(self, val)
            self.stub = None

    def ndimension(self):
        r"""
        Alias for :func:`mindspore.Tensor.ndim`.
        """
        return self.ndim

    def asnumpy(self):
        """api stub."""
        self.stub_sync()
        return super().asnumpy()

    def is_persistent_data(self):
        """
        For details, please refer to :`mindspore.common.tensor.is_persistent_data`.
        """
        self.stub_sync()
        super().is_persistent_data()

    def asnumpy_of_slice_persistent_data(self, param_key, slice_index):
        """
        For details, please refer to :`mindspore.common.tensor.asnumpy_of_slice_persistent_data`.
        """
        self.stub_sync()
        return super().asnumpy_of_slice_persistent_data(param_key, slice_index)

    def slice_num_of_persistent_data(self):
        """
        For details, please refer to :`mindspore.common.tensor.slice_num_of_persistent_data`.
        """
        self.stub_sync()
        return super().slice_num_of_persistent_data()

    def slice_shape_of_persistent_data(self):
        """
        For details, please refer to :`mindspore.common.tensor.slice_shape_of_persistent_data`.
        """
        self.stub_sync()
        return super().slice_shape_of_persistent_data()

    def flush_from_cache(self):
        """
        For details, please refer to :`mindspore.common.tensor.flush_from_cache`.
        """
        self.stub_sync()
        super().flush_from_cache()


class StubTuple(tuple):
    """tuple that may contain stub tensor for async op run."""

    def __new__(cls, stub_tuple):
        """Do some pre-process before creating StubTuple

        Args:
            stub_tuple (tuple): a tuple of c_expression object that may contain `StubNode`

        Returns:
            StubTuple: a tuple of python object, in which `StubNode` is converted to `StubTensor`
        """
        new_tuple = StubTuple._dfs_convert_stubnode(stub_tuple)
        ret = super(StubTuple, cls).__new__(cls, new_tuple)
        return ret

    @staticmethod
    def _is_c_expression_stubnode(node):
        """Currently we just simply use the function that defined in `py::class_<StubNode>`"""
        return hasattr(node, "get_value")

    @staticmethod
    def _dfs_convert_stubnode(node):
        if isinstance(node, (tuple, list)):
            res = [StubTuple._dfs_convert_stubnode(o) for o in node]
            return type(node)(res)
        if StubTuple._is_c_expression_stubnode(node):
            # Identify and handle CSR/COO/ROW Tensor here, we can use `_stub_map`
            return StubTensor(node)
        return node


_stub_map = [
    StubTensor,
    StubTensor,  # CSR_TENSOR
    StubTensor,  # COO_TENSOR
    StubTensor,  # ROW_TENSOR
    StubTuple,
]


def _convert_stub(stub_type, stub):
    """convert stub node to stub tensor."""
    if stub_type >= len(_stub_map):  # obj that already convert in c++, e.g. Scalar
        return stub
    return _stub_map[stub_type](stub)

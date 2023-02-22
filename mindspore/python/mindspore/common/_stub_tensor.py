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
from mindspore._c_expression import TensorNode, SequenceNode
from mindspore.common.api import _convert_python_data


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
        return False

    @property
    def adapter_flag(self):
        """adapter_flag stub."""
        if self.stub:
            return False
        return super().adapter_flag

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

    def dim(self):
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


def _convert_stub(stub):
    if isinstance(stub, TensorNode):
        return StubTensor(stub)
    if isinstance(stub, tuple):
        return tuple(_convert_stub(e) for e in stub)
    if isinstance(stub, SequenceNode):
        elements = stub.get_elements()
        return tuple(_convert_stub(e) for e in elements)
    return _convert_python_data(stub)

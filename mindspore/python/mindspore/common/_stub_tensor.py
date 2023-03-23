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

import inspect
from functools import reduce
from mindspore.common.tensor import Tensor
from mindspore.common.dtype import type_size_in_bytes
from mindspore._c_expression import TensorNode, SequenceNode, NoneTypeNode, AnyTypeNode
from mindspore._c_expression import Tensor as Tensor_
from mindspore.common.api import _convert_python_data


def _stub_member(var, init):
    """handle stub tensor's member, use a member cache to improve performance"""
    def getx(stub):
        if stub.tensor is not None:
            return getattr(stub.tensor, var)
        if hasattr(stub, "member_cache"):
            return getattr(stub.member_cache, var, init)
        return init

    def setx(stub, value):
        if stub.tensor is not None:
            setattr(stub.tensor, var, value)
        else:
            if not hasattr(stub, "member_cache"):
                stub.member_cache = {}
            stub.member_cache[var] = value
    return property(getx, setx)


def _stub_method(method):
    def fun(*arg, **kwargs):
        stub = arg[0]
        arg = (stub.stub_sync(),) + arg[1:]
        return method(*arg, **kwargs)
    return fun


class StubTensor:
    """stub tensor for async op run."""
    const_arg = _stub_member("const_arg", None)
    init = _stub_member("init", None)
    init_finished = _stub_member("init_finished", False)
    virtual_flag = _stub_member("virtual_flag", False)
    parent_tensor_ = _stub_member("parent_tensor_", None)
    index_of_parent_ = _stub_member("index_of_parent_", None)
    slice_num_of_persistent_data_ = _stub_member("slice_num_of_persistent_data_", None)
    slice_shape_of_persistent_data_ = _stub_member("slice_shape_of_persistent_data_", None)

    def __init__(self, stub):
        self.stub = stub
        self.tensor = None

    __str__ = _stub_method(Tensor.__str__)
    __setitem__ = _stub_method(Tensor.__setitem__)

    __lt__ = Tensor.__lt__
    __le__ = Tensor.__le__
    __gt__ = Tensor.__gt__
    __ge__ = Tensor.__ge__
    __eq__ = Tensor.__eq__
    __ne__ = Tensor.__ne__

    def __repr__(self):
        try:
            return _stub_method(Tensor.__repr__)(self)
        except RuntimeError:
            return "{} object at {}\nThe real tensor data is {} due to the internal error below.".format(
                type(self), hex(id(self)), type(self.tensor))

    @property
    def shape(self):
        """shape stub."""
        if self.stub:
            if not hasattr(self, "stub_shape"):
                self.stub_shape = self.stub.get_shape()
            return self.stub_shape
        return self.tensor.shape

    @property
    def dtype(self):
        """dtype stub."""
        if self.stub:
            if not hasattr(self, "stub_dtype"):
                self.stub_dtype = self.stub.get_dtype()
            return self.stub_dtype
        return self.tensor.dtype

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
    def adapter_flag(self):
        """adapter flag."""
        return False

    @property
    def strides(self):
        """strides stub."""
        return self.stub_sync().strides

    @property
    def has_init(self):
        """has_init stub."""
        return False

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

    asnumpy = _stub_method(Tensor.asnumpy)
    is_persistent_data = _stub_method(Tensor.is_persistent_data)
    asnumpy_of_slice_persistent_data = _stub_method(Tensor.asnumpy_of_slice_persistent_data)
    slice_num_of_persistent_data = _stub_method(Tensor.slice_num_of_persistent_data)
    slice_shape_of_persistent_data = _stub_method(Tensor.slice_shape_of_persistent_data)
    flush_from_cache = _stub_method(Tensor.flush_from_cache)

    def stub_sync(self):
        """Get value of a stubtensor."""
        if self.stub:
            val = self.stub.get_value()
            self.tensor = Tensor(val, internal=True)
            if hasattr(self, "member_cache"):
                for k, v in self.member_cache.items():
                    setattr(self.tensor, k, v)
            self.stub = None
        return self.tensor


def _init_stub_tensor_api():
    """adapt to python tensor and cpp tensor api"""
    need_init_func = set(dir(Tensor)) - set(dir(StubTensor))
    cpp_tensor_func = dir(Tensor_)
    for attr in need_init_func:
        func = inspect.getattr_static(Tensor, attr)
        if attr in cpp_tensor_func:
            # for cpp tensor api, we always need to sync for real tensor first
            setattr(StubTensor, attr, _stub_method(func))
        else:
            setattr(StubTensor, attr, func)


_init_stub_tensor_api()


def _convert_stub(stub):
    "convert stub to StubNode or Value"
    if isinstance(stub, TensorNode):
        return StubTensor(stub)
    if isinstance(stub, tuple):
        return tuple(_convert_stub(e) for e in stub)
    if isinstance(stub, SequenceNode):
        elements = stub.get_elements()
        return tuple(_convert_stub(e) for e in elements)
    if isinstance(stub, NoneTypeNode):
        val = stub.get_real_value()
        return _convert_python_data(val)
    if isinstance(stub, AnyTypeNode):
        val = stub.get_real_node()
        return _convert_stub(val)
    return _convert_python_data(stub)

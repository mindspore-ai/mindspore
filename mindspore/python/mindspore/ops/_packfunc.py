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
"""Context for PackFunc"""

import inspect
from functools import reduce
from mindspore import Tensor
from mindspore.ops.primitive import _RunOpHook, Primitive
from mindspore.common.dtype import type_size_in_bytes
from mindspore._c_expression import PackExpander, PackNode


def _unsupport_method(method):
    def fun(*arg, **kwargs):
        raise Exception("unsuport method call: " + str(method))
    return fun


class _PackTensor:
    """stub tensor for expander trace"""

    __repr__ = _unsupport_method(Tensor.__repr__)
    __str__ = _unsupport_method(Tensor.__str__)
    __setitem__ = _unsupport_method(Tensor.__setitem__)

    __lt__ = Tensor.__lt__
    __le__ = Tensor.__le__
    __gt__ = Tensor.__gt__
    __ge__ = Tensor.__ge__
    __eq__ = Tensor.__eq__
    __ne__ = Tensor.__ne__

    asnumpy = _unsupport_method(Tensor.asnumpy)
    is_persistent_data = _unsupport_method(Tensor.is_persistent_data)
    asnumpy_of_slice_persistent_data = _unsupport_method(Tensor.asnumpy_of_slice_persistent_data)
    slice_num_of_persistent_data = _unsupport_method(Tensor.slice_num_of_persistent_data)
    slice_shape_of_persistent_data = _unsupport_method(Tensor.slice_shape_of_persistent_data)
    flush_from_cache = _unsupport_method(Tensor.flush_from_cache)

    def __init__(self, node):
        self.pack_node = node

    @property
    def shape(self):
        """shape stub."""
        return self.pack_node.get_shape()

    @property
    def dtype(self):
        """dtype stub."""
        return self.pack_node.get_dtype()

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

    def stub_sync(self):
        """subclass hook for Tensor"""
        return self


def _init_trace_tensor_api():
    stub_func = dir(_PackTensor)
    for attr in dir(Tensor):
        if attr not in stub_func:
            func = inspect.getattr_static(Tensor, attr)
            setattr(_PackTensor, attr, func)


_init_trace_tensor_api()


def _convert_tensor(node):
    if isinstance(node, PackNode):
        return _PackTensor(node)
    if isinstance(node, tuple):
        return tuple(_convert_tensor(e) for e in node)
    return node


class PackFunc(Primitive):
    """pack function with lazy expander"""

    expander = PackExpander.get_instance()

    def __init__(self, fun, unique_key):
        super(PackFunc, self).__init__(self.__class__.__name__)
        self.func = fun
        self.add_prim_attr("unique_key", unique_key)

    def __expand__(self, args):
        with _RunOpHook(PackFunc._trace_run_op):
            fun_args = [_convert_tensor(a) for a in args]
            ret = self.func(*fun_args)
        return ret

    @staticmethod
    def _trace_run_op(obj, args):
        ret = PackFunc.expander.emit(obj, *args)
        return _convert_tensor(ret)


def pack(fn):
    """Create an pack func from a python function"""
    pack_func = PackFunc(fn, id(fn))
    def _pack_wrap(*args):
        if args and not isinstance(args[0], Tensor) and hasattr(args[0], fn.__name__):
            obj = args[0]
            return PackFunc(lambda *args_: fn(obj, *args_), id(fn))(*args[1:])
        return pack_func(*args)
    _pack_wrap.fn = fn
    return _pack_wrap

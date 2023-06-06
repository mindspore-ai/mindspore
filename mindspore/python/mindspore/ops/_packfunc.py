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
import functools
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import _RunOpHook, Primitive
from mindspore._c_expression import PackExpander, PackNode
from mindspore.common._stub_tensor import StubTensor
from mindspore.common._register_for_tensor import tensor_operator_registry


class _PackTensor(StubTensor):
    """stub tensor for expander trace"""

    def __setitem__(self, index, value):
        out = tensor_operator_registry.get('__setitem__')(self, index, value)
        self.stub = out.stub
        if self.parent_tensor_ is not None and self.index_of_parent_ is not None:
            self.parent_tensor_.__setitem__(self.index_of_parent_, self)
        return self

    def __pack__(self):
        """For parse check."""

    def stub_sync(self):
        """subclass hook for Tensor"""
        if self.tensor is None:
            val = self.stub.get_value()
            if val is None:
                raise Exception(
                    "In construct PackFunc, PackTensor has no real value.")
            self.tensor = Tensor(val, internal=True)
            if hasattr(self, "member_cache"):
                for k, v in self.member_cache.items():
                    setattr(self.tensor, k, v)
        return self.tensor


def _convert_tensor(node):
    if isinstance(node, PackNode):
        return _PackTensor(node)
    if isinstance(node, tuple):
        return tuple(_convert_tensor(e) for e in node)
    return node


class PackFunc(Primitive):
    """pack function with lazy expander"""

    expander = PackExpander.get_instance()

    def __init__(self, fun, unique_key, **kwarg):
        super(PackFunc, self).__init__(self.__class__.__name__)
        self.func = fun
        self.add_prim_attr("unique_key", unique_key)
        self.kwarg = kwarg

    def __call__(self, *args):
        if _RunOpHook.current and _RunOpHook.current.hook is PackFunc._trace_run_op:
            return self.func(*args, **self.kwarg)
        return super().__call__(*args)

    def __expand__(self, args):
        with _RunOpHook(PackFunc._trace_run_op):
            fun_args = [_convert_tensor(a) for a in args]
            ret = self.func(*fun_args, **self.kwarg)
        return ret

    @staticmethod
    def _trace_run_op(obj, args):
        ret = PackFunc.expander.emit(obj, *args)
        return _convert_tensor(ret)


def pack(fn):
    """Create an pack func from a python function"""

    @functools.wraps(fn)
    def _pack_wrap(*args, **kwarg):
        if args and not isinstance(args[0], Tensor) and hasattr(args[0], fn.__name__):
            obj = args[0]
            res = PackFunc(
                lambda *args_, **kwarg_: fn(obj, *args_, **kwarg_), id(fn), **kwarg)(*args[1:])
        else:
            res = PackFunc(fn, id(fn), **kwarg)(*args)
        return res
    _pack_wrap.pack_fn = fn
    return _pack_wrap

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
import types
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import _RunOpHook, Primitive
from mindspore._c_expression import PackExpander, PackNode
from mindspore.common._stub_tensor import StubTensor
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.common.api import _handle_func_args


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
                raise RuntimeError("During the trace operation, the data flow of the Tensor could be tracked, "
                                   "which consequently prevented the creation of a proper trace subgraph.")
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

    def __init__(self, fun, unique_key, cell_obj, is_pynative_mode=False):
        super(PackFunc, self).__init__(self.__class__.__name__)
        self.func = fun
        self.kwargs = {}
        self.add_prim_attr("unique_key", unique_key)
        self.add_prim_attr("is_pynative_mode", is_pynative_mode)
        self.cell_obj = cell_obj

    def __call__(self, *args, **kwargs):
        if _RunOpHook.current and _RunOpHook.current.hook is PackFunc._trace_run_op:
            if self.cell_obj is not None:
                args = (self.cell_obj, *args)
            return self.func(*args, **kwargs)
        self.kwargs = kwargs
        return super().__call__(*args)

    def __expand__(self, args):
        if self.cell_obj is not None:
            args = (self.cell_obj, *args)
            with _SetMixedPrecision(self.cell_obj):
                ret = self._run_op(args)
            return ret
        return self._run_op(args)

    @staticmethod
    def _trace_run_op(obj, args):
        ret = PackFunc.expander.emit(obj, *args)
        return _convert_tensor(ret)

    def _run_op(self, args):
        with _RunOpHook(PackFunc._trace_run_op):
            fun_args = [_convert_tensor(a) for a in args]
            ret = self.func(*fun_args, **self.kwargs)
        return ret


class _PackSourceBuilder:
    """Generation Pack Python code by method"""

    def __init__(self, original_fn):
        self.original_fn = original_fn
        self.pack_fn_name = f"_{original_fn.__name__}_pack"
        self._generate_pack_op()

    def get_code_source(self):
        """Return Pack Python code"""
        if isinstance(self.original_fn, types.MethodType):
            new_src = "def {0}_wrap(self, *args, **kwargs):\n    return self.{0}(*args, **kwargs)".format(
                self.pack_fn_name)
        else:
            new_src = "def {0}_wrap(*args, **kwargs):\n    return {0}(*args, **kwargs)".format(
                self.pack_fn_name)
        return new_src

    def _generate_pack_op(self):
        fn = self.original_fn.pack_fn
        if isinstance(self.original_fn, types.MethodType):
            obj = self.original_fn.__self__
            key = "%d_ID%d" % (id(obj), id(fn))
            setattr(obj, self.pack_fn_name, PackFunc(fn, key, obj))
        else:
            key = str(id(fn))
            fn.__globals__[self.pack_fn_name] = PackFunc(fn, key, None)


class _SetMixedPrecision:
    """"Set MixedPrecison by the Cell"""
    def __init__(self, cell_obj):
        self.mixed_precision_change = False
        self.cell_obj = cell_obj

    def __enter__(self):
        self.mixed_precision_change = PackFunc.expander.set_mixed_precision(self.cell_obj)

    def __exit__(self, *err):
        if self.mixed_precision_change:
            PackFunc.expander.recover_mixed_precision()


def pack(fn):
    """Create an pack func from a python function"""

    @functools.wraps(fn)
    def _pack_wrap(*args, **kwargs):
        args, kwargs = _handle_func_args(fn, *args, **kwargs)
        if args and not isinstance(args[0], Tensor) and hasattr(args[0], fn.__name__):
            obj = args[0]
            key = "%d_ID%d" % (id(obj), id(fn))
            res = PackFunc(fn, key, obj, True)(*args[1:], **kwargs)
        else:
            key = str(id(fn))
            res = PackFunc(fn, key, None, True)(*args, **kwargs)
        return res
    _pack_wrap.pack_fn = fn
    return _pack_wrap

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
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import _RunOpHook, Primitive
from mindspore._c_expression import PackExpander, PackNode
from mindspore.common._stub_tensor import StubTensor


class _PackTensor(StubTensor):
    """stub tensor for expander trace"""

    def stub_sync(self):
        """subclass hook for Tensor"""
        if self.tensor is None:
            val = self.stub.get_value()
            if val is None:
                raise Exception("In construct PackFunc, PackTensor has no real value.")
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


_PACK_EXECUTING = False


def pack(fn):
    """Create an pack func from a python function"""
    pack_func = PackFunc(fn, id(fn))

    def _pack_wrap(*args):
        global _PACK_EXECUTING
        if _PACK_EXECUTING:
            return fn(*args)
        _PACK_EXECUTING = True
        try:
            if args and not isinstance(args[0], Tensor) and hasattr(args[0], fn.__name__):
                obj = args[0]
                res = PackFunc(lambda *args_: fn(obj, *args_),
                               id(fn))(*args[1:])
            else:
                res = pack_func(*args)
        finally:
            _PACK_EXECUTING = False
        return res
    _pack_wrap.fn = fn
    return _pack_wrap

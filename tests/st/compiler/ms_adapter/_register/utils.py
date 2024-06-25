# Copyright 2022 Huawei Technologies Co., Ltd
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

import mindspore as ms
from mindspore import dtype as mstype
from mindspore._c_expression import typing
from mindspore.ops.operations import _inner_ops as inner
from tests.st.compiler.ms_adapter._register.ms_adapter_api import Tensor as adapter_Tensor


def convert_to_ms_tensor(x):
    return inner.convert_to_ms_tensor(x)


def convert_to_adapter_tensor(x):
    return inner.convert_to_adapter_tensor(x)


def get_registed_fn(ops, *type_names):
    types = tuple(map(mstype.typing.str_to_type, type_names))
    for sigs, fn in ops.entries:
        if len(sigs) != len(types):
            continue
        if any(not typing.is_subclass(type_, sig) for sig, type_ in zip(sigs, types)):
            continue
        return fn
    raise ValueError(f"For 'MultitypeFuncGraph', cannot find fn match given types: {types}.")


def convert_output(out):
    if isinstance(out, ms.Tensor):
        out = convert_to_adapter_tensor(out)
    return out


def update_multitype_ops_tensor(ops):
    func = get_registed_fn(ops, "Tensor")

    @ops.register("Tensor")
    def _tensor(x):
        if isinstance(x, adapter_Tensor):
            x = convert_to_ms_tensor(x)
            out = func(x)
            out = convert_output(out)
        else:
            out = func(x)
        return out


def update_multitype_ops_tensor_tensor(ops):
    func = get_registed_fn(ops, "Tensor", "Tensor")

    @ops.register("Tensor", "Tensor")
    def _tensor_and_tensor(x, y):
        if isinstance(x, adapter_Tensor) and isinstance(y, adapter_Tensor):
            x = convert_to_ms_tensor(x)
            y = convert_to_ms_tensor(y)
            out = func(x, y)
            out = convert_output(out)
        else:
            out = func(x, y)
        return out


def update_multitype_ops_number_tensor(ops):
    func = get_registed_fn(ops, "Number", "Tensor")

    @ops.register("Number", "Tensor")
    def _number_and_tensor(x, y):
        if isinstance(y, adapter_Tensor):
            y = convert_to_ms_tensor(y)
            out = func(x, y)
            out = convert_output(out)
        else:
            out = func(x, y)
        return out


def update_multitype_ops_tensor_number(ops):
    func = get_registed_fn(ops, "Tensor", "Number")

    @ops.register("Tensor", "Number")
    def _tensor_and_number(x, y):
        if isinstance(x, adapter_Tensor):
            x = convert_to_ms_tensor(x)
            out = func(x, y)
            out = convert_output(out)
        else:
            out = func(x, y)
        return out


def update_multitype_ops_tuple_tensor(ops):
    func = get_registed_fn(ops, "Tuple", "Tensor")

    @ops.register("Tuple", "Tensor")
    def _tuple_and_tensor(x, y):
        if isinstance(y, adapter_Tensor):
            y = convert_to_ms_tensor(y)
            out = func(x, y)
            out = convert_output(out)
        else:
            out = func(x, y)
        return out


def update_multitype_ops_tensor_tuple(ops):
    func = get_registed_fn(ops, "Tensor", "Tuple")

    @ops.register("Tensor", "Tuple")
    def _tensor_and_tuple(x, y):
        if isinstance(x, adapter_Tensor):
            x = convert_to_ms_tensor(x)
            out = func(x, y)
            out = convert_output(out)
        else:
            out = func(x, y)
        return out


def update_multitype_ops_list_tensor(ops):
    func = get_registed_fn(ops, "List", "Tensor")

    @ops.register("List", "Tensor")
    def _list_and_tensor(x, y):
        if isinstance(y, adapter_Tensor):
            y = convert_to_ms_tensor(y)
            out = func(x, y)
            out = convert_output(out)
        else:
            out = func(x, y)
        return out


def update_multitype_ops_tensor_list(ops):
    func = get_registed_fn(ops, "Tensor", "List")

    @ops.register("Tensor", "List")
    def _tensor_and_list(x, y):
        if isinstance(x, adapter_Tensor):
            x = convert_to_ms_tensor(x)
            out = func(x, y)
            out = convert_output(out)
        else:
            out = func(x, y)
        return out


def update_multitype_ops_tensor_none(ops):
    func = get_registed_fn(ops, "Tensor", "None")

    @ops.register("Tensor", "None")
    def _tensor_and_none(x, y):
        if isinstance(x, adapter_Tensor):
            x = convert_to_ms_tensor(x)
            out = func(x, y)
            out = convert_output(out)
        else:
            out = func(x, y)
        return out


def update_multitype_ops_tensor_slice(ops):
    func = get_registed_fn(ops, "Tensor", "Slice")

    @ops.register("Tensor", "Slice")
    def _tensor_and_slice(x, y):
        if isinstance(x, adapter_Tensor):
            x = convert_to_ms_tensor(x)
            out = func(x, y)
            out = convert_output(out)
        else:
            out = func(x, y)
        return out


def update_multitype_ops_setitem_tensor(ops):
    def register_for_setitem(sigs, fn):
        @ops.register(*sigs)
        def _tensor_setitem(data, index, value):
            if isinstance(data, adapter_Tensor):
                data = convert_to_ms_tensor(data)
                out = fn(data, index, value)
                out = convert_to_adapter_tensor(out)
            else:
                out = fn(data, index, value)
            return out

    entries = ops.entries.copy()
    for sigs, fn in entries:
        if typing.is_subclass(sigs[0], mstype.tensor_type):
            register_for_setitem(sigs, fn)

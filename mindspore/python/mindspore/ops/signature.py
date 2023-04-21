# Copyright 2020 Huawei Technologies Co., Ltd
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

"""signature"""

from .._c_expression import signature_rw as sig_rw
from .._c_expression import signature_kind as sig_kind
from .._c_expression import signature_dtype as sig_dtype
from .._c_expression import Signature


def make_sig(name="var", rw=sig_rw.RW_READ,
             kind=sig_kind.KIND_POSITIONAL_KEYWORD,
             default=sig_kind.KIND_EMPTY_DEFAULT_VALUE,
             dtype=sig_dtype.T_EMPTY_DEFAULT_VALUE):
    """
    Make signature for one argument.

    See `ApplyMomentum` in `mindspore.ops.operation.nn_ops` as a example.

    Args:
         name (bool): Argument name. Default: ``"var"`` .
         rw (:class:`mindspore.ops.signature.sig_rw`): Tag the argument attribute for write and read. Choose in
            [sig_rw.RW_READ, sig_rw.RW_WRITE, sig_rw.RW_REF]`, tag if the argument will update the input.
            `sig_rw.RW_READ` for read only argument and `sig_rw.RW_WRITE` for write only argument. `sig_rw.RW_READ`
            for the argument both need read and write. Default: ``sig_rw.RW_READ`` .
         kind (:class:`mindspore.ops.signature.kind`): Choose in `[signature_kind.KIND_POSITIONAL_KEYWORD,
            signature_kind.KIND_VAR_POSITIONAL, signature_kind.KIND_KEYWORD_ONLY, signature_kind.KIND_VAR_KEYWARD]`.
            The meaning is the same as python argument kind, please refer to the python document.
            Default: ``sig_kind.KIND_POSITIONAL_KEYWORD`` .
         default (Any): The default value of argument or `sig_kind.KIND_EMPTY_DEFAULT_VALUE` for no default value.
            Default: ``sig_kind.KIND_EMPTY_DEFAULT_VALUE`` .
         dtype (:class:`mindspore.ops.signature.sig_dtype`): Choose in `signature_dtype.T` or
            `signature_dtype.T1` to `signature_dtype.T9` or `sig_dtype.T_EMPTY_DEFAULT_VALUE` for no constraints.
            If the signature of one argument is the same as another argument, we will perform auto type convert
            between them. If any `sig_rw.RW_WRITE` argument, we will try to convert the other arguments to the
            `sig_rw.RW_WRITE` argument. Default: ``sig_dtype.T_EMPTY_DEFAULT_VALUE`` .

    Returns:
        :class:`mindspore.ops.signature.Signature`, signature for one argument.
    """
    return Signature(name, rw, kind, default, dtype)

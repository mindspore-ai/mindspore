# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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
"""Trope some system function symbol to ops."""

# This operation function is not meant to be called directly

# support operator symbol, ast
from operator import (  # noqa
    add, sub, mul, truediv, floordiv, mod, eq, ne, lt, gt, le, ge, pos, neg,
    not_, and_, or_, xor, lshift, rshift, invert, is_, is_not, contains,
    matmul, getitem, setitem
)

# support system function call
from builtins import (  # noqa
    bool, getattr, setattr, len, iter, next, pow, range, map, zip, print, enumerate, isinstance
)

# support functools
from functools import (  # noqa
    partial
)

# support numpy symbol
from numpy import (  # noqa
    exp, log, sin, cos, tan
)

__all__ = ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'eq', 'ne', 'lt', 'gt', 'le', 'ge', 'pos', 'neg',
           'not_', 'and_', 'or_', 'xor', 'lshift', 'rshift', 'invert', 'is_', 'is_not', 'contains',
           'matmul', 'getitem', 'setitem',
           'bool', 'getattr', 'setattr', 'len', 'iter', 'next', 'pow', 'range', 'map', 'zip',
           'partial', 'print', 'enumerate', 'isinstance',
           'exp', 'log', 'sin', 'cos', 'tan']


def MakeTuple(*elts):  # pragma: no cover
    """Tuple builder."""
    raise RuntimeError('This operation is not meant to be called directly.')


def make_dict(key, value):  # pragma: no cover
    """Dict builder."""
    raise RuntimeError('This operation is not meant to be called directly.')


def make_list(*elts):  # pragma: no cover
    """List builder."""
    raise RuntimeError('This operation is not meant to be called directly.')


def make_slice(*elts):  # pragma: no cover
    """Slice builder."""
    raise RuntimeError('This operation is not meant to be called directly.')


def make_range(*elts):  # pragma: no cover
    """Range tuple builder."""
    raise RuntimeError('This operation is not meant to be called directly.')


def switch(cond, tb, fb):  # pragma: no cover
    """Switch statement, returns one of the two values."""
    raise RuntimeError('This operation is not meant to be called directly.')


def hasnext(it):  # pragma: no cover
    """Hasnext function."""
    raise RuntimeError('This operation is not meant to be called directly.')


def to_array(x):  # pragma: no cover
    """The to_array function."""
    raise RuntimeError('This operation is not meant to be called directly.')


def not_contains(x):  # pragma: no cover
    """Not in function."""
    raise RuntimeError('This operation is not meant to be called directly.')

def while_cond(x):  # pragma: no cover
    """Not in function."""
    raise RuntimeError('This operation is not meant to be called directly.')

def bool_(x):  # pragma: no cover
    """judge true function."""
    raise RuntimeError('This operation is not meant to be called directly.')

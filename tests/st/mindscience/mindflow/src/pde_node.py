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
"""
construct nodes for sympy expressions
"""

import functools as ft

import numpy as np

import mindspore.numpy as mnp
from mindspore import dtype as mstype
from mindspore import ops, jit_class, Tensor


def _reduce(fn):
    def fn_(*args):
        return ft.reduce(fn, args)

    return fn_


MINDSPORE_SYMPY_TRANSLATIONS = {
    "Mul": _reduce(mnp.multiply),
    "Add": _reduce(mnp.add),
    "div": mnp.divide,
    "Abs": mnp.abs,
    "sign": mnp.sign,
    "ceiling": mnp.ceil,
    "floor": mnp.floor,
    "log": mnp.log,
    "exp": mnp.exp,
    "sqrt": mnp.sqrt,
    "cos": mnp.cos,
    "acos": mnp.arccos,
    "sin": mnp.sin,
    "asin": mnp.arcsin,
    "tan": mnp.tan,
    "atan": mnp.arctan,
    "atan2": mnp.arctan2,
    "cosh": mnp.cosh,
    "acosh": mnp.arccosh,
    "sinh": mnp.sinh,
    "asinh": mnp.arcsinh,
    "tanh": mnp.tanh,
    "atanh": mnp.arctanh,
    "Pow": mnp.power,
    "re": ops.Real,
    "im": ops.Imag,
    "arg": np.angle,
    "erf": ops.erf,
    "Eq": mnp.equal,
    "Ne": mnp.not_equal,
    "StrictGreaterThan": mnp.greater,
    "StrictLessThan": mnp.less,
    "LessThan": mnp.less_equal,
    "GreaterThan": mnp.greater_equal,
    "And": mnp.logical_and,
    "Or": mnp.logical_or,
    "Not": mnp.logical_not,
    "Xor": mnp.logical_xor,
    "Max": _reduce(mnp.maximum),
    "Min": _reduce(mnp.minimum),
    "Trace": mnp.trace,
}


@jit_class
class AddNode:
    """Compute add terms in sympy expression"""

    def __init__(self, nodes=None):
        self.nodes = nodes or list()

    def compute(self, data):
        rst = list()
        for node in self.nodes:
            cur_node_rst = node.compute(data)
            rst.append(cur_node_rst)

        return sum(rst)


@jit_class
class PowNode:
    """Compute pow terms in sympy expression"""

    def __init__(self, nodes=None):
        self.nodes = nodes or list()

    def compute(self, data):
        return ops.pow(self.nodes[0].compute(data), self.nodes[1].compute(data))


@jit_class
class MulNode:
    """Compute multiplication terms in sympy expression"""

    def __init__(self, nodes=None):
        self.nodes = nodes or list()

    def compute(self, data):
        """compute the result of mul expression"""
        rst = Tensor(np.float32(1.0), mstype.float32)
        for node in self.nodes:
            rst = rst * node.compute(data)
        return rst


@jit_class
class NumberNode:
    """Compute number terms in sympy expression"""

    def __init__(self, nodes=None):
        self.nodes = nodes or list()

    def compute(self, data):
        """compute the result of number"""
        if not isinstance(data, dict):
            raise TypeError("For 'compute', only dict data is supported.")
        return self.nodes[0]


@jit_class
class SymbolNode:
    """Compute symbol terms in sympy expression"""

    def __init__(self, in_vars, in_var_idx=None):
        self.input_split = ops.Split(1, len(in_vars))
        self.in_var_idx = in_var_idx

    def compute(self, data):
        """compute the result of symbol"""
        input_data = data.get("inputs")
        ret = self.input_split(input_data)[self.in_var_idx]
        return ret


@jit_class
class ParamNode:
    """Compute parameter terms in sympy expression"""

    def __init__(self, params, param_var_idx=None):
        self.param_split = ops.Split(-1, len(params))
        self.param_var_idx = param_var_idx

    def compute(self, data):
        """compute the result of parameter"""
        params_data = data.get("params")
        ret = self.param_split(params_data)[self.param_var_idx]
        return ret


@jit_class
class NetOutputNode:
    """Compute network function terms in sympy expression"""

    def __init__(self, out_vars, out_var_idx=None):
        self.output_split = ops.Split(1, len(out_vars))
        self.out_var_idx = out_var_idx

    def compute(self, data):
        """compute the result of network"""
        output_data = data.get("outputs")
        ret = self.output_split(output_data)[self.out_var_idx]
        return ret


@jit_class
class MSFunctionNode:
    """Compute function which can be translated into mindspore function in sympy expression"""

    def __init__(self, nodes=None, fn=None):
        self.nodes = nodes or list()
        self.fn = fn

    def compute(self, data):
        """compute the result of mindspore function"""
        ret = self.fn(self.nodes[0].compute(data))
        return ret


@jit_class
class DerivativeNode:
    """Compute derivative terms in sympy expression"""

    def __init__(self, in_vars, order=None, in_var_idx=None, out_var_idx=None, is_norm=False):
        self.input_split = ops.Split(1, len(in_vars))
        self.order = order
        self.in_var_idx = in_var_idx
        self.out_var_idx = out_var_idx
        self.is_norm = is_norm

    def compute(self, data):
        """compute the result of derivative expression"""
        if self.order == 1:
            jacobian = data.get("jacobian")
            derivative_out = jacobian[self.out_var_idx]
            if self.is_norm:
                norm = data.get("norm")
                if norm.ndim == 1:
                    ret = ops.matmul(derivative_out, norm)
                else:
                    ret = (derivative_out * norm).sum(axis=1)
            else:
                ret = self.input_split(derivative_out)[self.in_var_idx]
            return ret

        if self.order == 2:
            hessian = data.get("hessian")
            derivative_out = hessian[self.out_var_idx][self.in_var_idx[0]]
            ret = self.input_split(derivative_out)[self.in_var_idx[1]]
            return ret

        raise ValueError("For `Derivative`, only first-order and second-order differentials are supported \
            but got {}".format(self.order))

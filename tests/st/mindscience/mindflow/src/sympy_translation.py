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
translate sympy expressions into mindspore recognized attribute.
"""

import numpy as np
import sympy
from mindspore import Tensor, jit_class
from mindspore import dtype as mstype

from .pde_node import MINDSPORE_SYMPY_TRANSLATIONS, MulNode, NumberNode, SymbolNode, ParamNode
from .pde_node import MSFunctionNode, NetOutputNode, DerivativeNode, PowNode, AddNode


@jit_class
class SympyTranslation:
    '''translate sympy expressions'''

    def __init__(self, formula, formula_node, in_vars, out_vars, params=None):
        self.formula = formula
        self.formula_node = formula_node
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.params = params
        print(f"{self.formula_node.name}: {self.formula}")
        self._parse_node()
        print(
            f"    Item numbers of current derivative formula nodes: {len(self.formula_node.nodes)}")

    @staticmethod
    def _check_item_args(item_args):
        """check item args"""
        if len(item_args) > 1:
            raise ValueError("For a composite function, only 1 args is supported for the function parameter, \
                but got {}.".format(len(item_args)))

    @staticmethod
    def _parse_number(item):
        """parse number"""
        nodes = [Tensor(np.float32(item), mstype.float32)]
        number_node = NumberNode(nodes=nodes)
        return number_node

    @staticmethod
    def _parse_number_symbol(item):
        """parses number symbol"""
        if isinstance(item, sympy.core.numbers.Pi):
            nodes = [Tensor(np.float32(np.pi), mstype.float32)]
            number_symbol_node = NumberNode(nodes=nodes)
        elif isinstance(item, sympy.core.numbers.Exp1):
            nodes = [Tensor(np.float32(np.exp(1), mstype.float32))]
            number_symbol_node = NumberNode(nodes=nodes)
        else:
            raise ValueError(
                "For NumberSymbol, sympy.pi and sympy.E are supported, but got {}.".format(item))

        return number_symbol_node

    def _parse_node(self):
        """parses each node in sympy expression"""
        if self.formula.is_Add:
            for item in self.formula.args:
                cur_node = self._parse_item(item)
                self.formula_node.nodes.append(cur_node)

        else:
            cur_node = self._parse_item(self.formula)
            self.formula_node.nodes.append(cur_node)

    def _parse_item(self, item):
        """parse each item in sympy expression"""
        if item.is_Mul:
            node = self._parse_mul(item)

        elif item.is_Number:
            node = self._parse_number(item)

        elif item.is_Symbol:
            node = self._parse_symbol(item)

        elif item.is_Function:
            node = self._parse_function(item)

        elif item.is_Derivative:
            node = self._parse_derivative(item)

        elif item.is_NumberSymbol:
            node = self._parse_number_symbol(item)

        elif item.is_Pow:
            node = self._parse_pow(item)

        else:
            raise ValueError(
                "For parsing sympy expression: {} is not supported!".format(item))

        return node

    def _parse_derivative(self, item):
        """parses derivative expression"""
        order = np.int8(0)
        for it in item.args[1:]:
            order += np.int8(it[1])

        self.formula_node.max_order = max(order, self.formula_node.max_order)
        # index of output vars
        out_var_idx = self.out_vars.index(item.args[0])

        if order == 1:
            cur_var = item.args[1][0]
            if cur_var == sympy.Symbol('n'):
                derivative_node = DerivativeNode(
                    self.in_vars, order=order, out_var_idx=out_var_idx, is_norm=True)
            else:
                # index of input vars
                in_var_idx = self.in_vars.index(cur_var)
                derivative_node = DerivativeNode(self.in_vars, order=order, in_var_idx=in_var_idx,
                                                 out_var_idx=out_var_idx)
        elif order == 2:
            var_idx = list()
            for it in item.args[1:]:
                for _ in range(it[1]):
                    in_var_idx = self.in_vars.index(it[0])
                    var_idx.append(in_var_idx)
            derivative_node = DerivativeNode(
                self.in_vars, order=order, in_var_idx=var_idx, out_var_idx=out_var_idx)
        else:
            raise ValueError("For `Derivative`, only first-order and second-order differentials are supported \
                but got {}".format(order))
        return derivative_node

    def _parse_function(self, item):
        """parses function"""
        if type(item).__name__ in MINDSPORE_SYMPY_TRANSLATIONS:
            function_node = self._parse_basic_ops(item)
        else:
            function_node = self._parse_network_function(item)
        return function_node

    def _parse_network_function(self, item):
        """parse network output function"""
        out_var_idx = self.out_vars.index(item)
        function_node = NetOutputNode(self.out_vars, out_var_idx=out_var_idx)
        return function_node

    def _parse_basic_ops(self, item):
        """parse the sympy expression which in mindspore ops"""
        fn = MINDSPORE_SYMPY_TRANSLATIONS.get(type(item).__name__)
        self._check_item_args(item.args)
        if item.args[0].is_Add:
            nodes = []
            for cur_item in item.args[0].args:
                cur_node = self._parse_item(cur_item)
                nodes.append(cur_node)
            fn_inside_node = AddNode(nodes=nodes)
        else:
            fn_inside_node = self._parse_item(item.args[0])
        function_node = MSFunctionNode(nodes=[fn_inside_node], fn=fn)

        return function_node

    def _parse_symbol(self, item):
        """parse symbol"""
        if item in self.in_vars:
            in_var_idx = self.in_vars.index(item)
            symbol_node = SymbolNode(self.in_vars, in_var_idx=in_var_idx)
        elif item in self.params:
            para_var_idx = self.params.index(item)
            symbol_node = ParamNode(self.params, param_var_idx=para_var_idx)
        else:
            raise ValueError(
                "Inputs and Parameters are supported, but got {}".format(item))
        return symbol_node

    def _parse_mul(self, item):
        """parse mul"""
        nodes = []
        for cur_item in item.args:
            cur_node = self._parse_item(cur_item)
            nodes.append(cur_node)
        mul_node = MulNode(nodes=nodes)
        return mul_node

    def _parse_pow(self, item):
        """parse pow"""
        nodes = []
        for cur_item in item.args:
            cur_node = self._parse_item(cur_item)
            nodes.append(cur_node)
        pow_node = PowNode(nodes=nodes)
        return pow_node

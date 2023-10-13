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
parse sympy equations
"""
import sympy

from mindspore import jit_class

from .sympy_translation import SympyTranslation


@jit_class
class FormulaNode:
    """
    The root node for sympy expression.

    Args:
         eq_name (str): the name of sympy expression.
    """

    def __init__(self, eq_name):
        self.name = eq_name
        self.nodes = list()
        self.max_order = 0

    def compute(self, data):
        rst = list()
        for node in self.nodes:
            cur_node_rst = node.compute(data)
            rst.append(cur_node_rst)

        return sum(rst)


def _make_nodes(equations, in_vars, out_vars, params=None):
    graph_nodes = list()
    for name, formula in equations.items():
        formula_node = FormulaNode(name)
        SympyTranslation(sympy.expand(formula), formula_node,
                         in_vars, out_vars, params)
        graph_nodes.append(formula_node)

    return graph_nodes


def sympy_to_mindspore(equations, in_vars, out_vars, params=None):
    """
    The sympy expression to create an identifier for mindspore.

    Args:
        equations (dict): the item in equations contains the key defined by user and the value is sympy expression.
        in_vars (list[sympy.core.Symbol]): list of all input variable symbols, consistent with the dimension of the
            input data.
        out_vars (list[sympy.core.Function]): list of all output variable symbols, consistent with the dimension of the
            output data.
        params (list[sympy.core.Function]): list of all parameter variable symbols.
    Returns:
        List([FormulaNode]), list of expressions node can be identified by mindspore.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.pde import sympy_to_mindspore
        >>> from sympy import symbols, Function, diff
        >>> x, y = symbols('x, y')
        >>> u = Function('u')(x, y)
        >>> in_vars = [x, y]
        >>> out_vars = [u]
        >>> eq1 = x + y
        >>> eq2 = diff(u, (x, 1)) + diff(u, (y, 1))
        >>> equations = {"eq1": eq1, "eq2": eq2}
        >>> res = sympy_to_mindspore(equations, in_vars, out_vars)
        >>> print(len(res))
        eq1: x + y
            Item numbers of current derivative formula nodes: 2
        eq2: Derivative(u(x, y), x) + Derivative(u(x, y), y)
            Item numbers of current derivative formula nodes: 2
        2
    """
    converted_equations = _make_nodes(equations, in_vars, out_vars, params)
    return converted_equations

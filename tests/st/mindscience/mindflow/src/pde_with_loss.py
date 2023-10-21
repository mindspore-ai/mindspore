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
Base class of user-defined pde problems.
"""
from sympy import diff, Function, symbols, Symbol
import numpy as np
from mindspore import jit_class

from .parse_sympy import sympy_to_mindspore
from .derivative import batched_hessian, batched_jacobian
from .utils import get_loss_metric


@jit_class
class PDEWithLoss:
    """
    Base class of user-defined pde problems.
    All user-defined problems to set constraint on each dataset should be inherited from this class.
    It is utilized to establish the mapping between each sub-dataset and used-defined loss functions.
    The loss will be calculated automatically by the constraint type of each sub-dataset. Corresponding member functions
    must be out_channels by user based on the constraint type in order to obtain the target label output. For example,
    for dataset1 the constraint type is "pde", so the member function "pde" must be overridden to tell that how to get
    the pde residual. The data(e.g. inputs) used to solve the residuals is passed to the parse_node, and the residuals
    of each equation can be automatically calculated.

    Args:
        model (mindspore.nn.Cell): Network for training.
        in_vars (List[sympy.core.Symbol]): Input variables of the `model`, represented by the sympy symbol.
        out_vars (List[sympy.core.Function]): Output variables of the `model`, represented by the sympy function.
        params (List[sympy.core.Function]): Parameters of the `model`, represented by the sympy function.
        params_val (List[sympy.core.Function]): Values of the Parameters from optimizer.

    Note:
        - The member function, "pde", must be overridden to define the symbolic derivative equqtions based on sympy.
        - The member function, "get_loss", must be overridden to caluate the loss of symbolic derivative equqtions.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.pde import PDEWithLoss, sympy_to_mindspore
        >>> from mindspore import nn, ops, Tensor
        >>> from mindspore import dtype as mstype
        >>> from sympy import symbols, Function, diff
        >>> class Net(nn.Cell):
        ...     def __init__(self, cin=2, cout=1, hidden=10):
        ...         super().__init__()
        ...         self.fc1 = nn.Dense(cin, hidden)
        ...         self.fc2 = nn.Dense(hidden, hidden)
        ...         self.fcout = nn.Dense(hidden, cout)
        ...         self.act = ops.Tanh()
        ...
        ...     def construct(self, x):
        ...         x = self.act(self.fc1(x))
        ...         x = self.act(self.fc2(x))
        ...         x = self.fcout(x)
        ...         return x
        >>> model = Net()
        >>> class MyProblem(PDEWithLoss):
        ...     def __init__(self, model, loss_fn=nn.MSELoss()):
        ...         self.x, self.y = symbols('x t')
        ...         self.u = Function('u')(self.x, self.y)
        ...         self.in_vars = [self.x, self.y]
        ...         self.out_vars = [self.u]
        ...         super(MyProblem, self).__init__(model, in_vars=self.in_vars, out_vars=self.out_vars)
        ...         self.loss_fn = loss_fn
        ...         self.bc_nodes = sympy_to_mindspore(self.bc(), self.in_vars, self.out_vars)
        ...
        ...     def pde(self):
        ...         my_eq = diff(self.u, (self.x, 2)) + diff(self.u, (self.y, 2)) - 4.0
        ...         equations = {"my_eq": my_eq}
        ...         return equations
        ...
        ...     def bc(self):
        ...         bc_eq = diff(self.u, (self.x, 1)) + diff(self.u, (self.y, 1)) - 2.0
        ...         equations = {"bc_eq": bc_eq}
        ...         return equations
        ...
        ...     def get_loss(self, pde_data, bc_data):
        ...         pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        ...         pde_loss = self.loss_fn(pde_res[0], Tensor(np.array([0.0]), mstype.float32))
        ...         bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
        ...         bc_loss = self.loss_fn(bc_res[0], Tensor(np.array([0.0]), mstype.float32))
        ...         return pde_loss + bc_loss
        >>> problem = MyProblem(model)
        >>> print(problem.pde())
        >>> print(problem.bc())
        my_eq: Derivative(u(x, t), (t, 2)) + Derivative(u(x, t), (x, 2)) - 4.0
            Item numbers of current derivative formula nodes: 3
        bc_eq: Derivative(u(x, t), t) + Derivative(u(x, t), x) - 2.0
            Item numbers of current derivative formula nodes: 3
        {'my_eq': Derivative(u(x, t), (t, 2)) + Derivative(u(x, t), (x, 2)) - 4.0}
        {'bc_eq': Derivative(u(x, t), t) + Derivative(u(x, t), x) - 2.0}
    """

    def __init__(self, model, in_vars, out_vars, params=None, params_val=None):
        self.model = model
        self.jacobian = batched_jacobian(self.model)
        self.hessian = batched_hessian(self.model)
        self.param_val = params_val
        pde_nodes = self.pde() or dict()
        if isinstance(pde_nodes, dict) and pde_nodes:
            self.pde_nodes = sympy_to_mindspore(
                pde_nodes, in_vars, out_vars, params)

    def pde(self):
        """
        Governing equation based on sympy, abstract method.
        This function must be overridden, if the corresponding constraint is governing equation.
        """
        return None

    def get_loss(self):
        """
        Compute all loss from user-defined derivative equations. This function must be overridden.
        """
        return None

    def parse_node(self, formula_nodes, inputs=None, norm=None):
        """
        Calculate the results for each formula node.

        Args:
            formula_nodes (list[FormulaNode]): List of expressions node can be identified by mindspore.
            inputs (Tensor): The input data of network. Default: ``None``.
            norm (Tensor): The normal of the surface at a point P is a vector perpendicular to the tangent plane of the
                point. Default: ``None``.

        Returns:
            List(Tensor), the results of the partial differential equations.
        """
        max_order = 0
        for formula_node in formula_nodes:
            max_order = max(formula_node.max_order, max_order)

        outputs = self.model(inputs)
        if max_order == 2:
            hessian = self.hessian(inputs)
            jacobian = self.jacobian(inputs)
        elif max_order == 1:
            hessian = None
            jacobian = self.jacobian(inputs)
        else:
            hessian = None
            jacobian = None

        if self.param_val is None:
            data_map = {"inputs": inputs, "outputs": outputs,
                        "jacobian": jacobian, "hessian": hessian, "norm": norm}
        else:
            data_map = {"inputs": inputs, "outputs": outputs, "jacobian": jacobian, "hessian": hessian,
                        "norm": norm, "params": self.param_val}
        res = []
        for formula_node in formula_nodes:
            cur_eq_ret = formula_node.compute(data_map)
            res.append(cur_eq_ret)
        return res


class Burgers(PDEWithLoss):
    r"""
    Base class for Burgers 1-D problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): Network for training.
        loss_fn (Union[str, Cell]): Define the loss function. Default: ``"mse"``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.pde import Burgers
        >>> from mindspore import nn, ops
        >>> class Net(nn.Cell):
        ...     def __init__(self, cin=2, cout=1, hidden=10):
        ...         super().__init__()
        ...         self.fc1 = nn.Dense(cin, hidden)
        ...         self.fc2 = nn.Dense(hidden, hidden)
        ...         self.fcout = nn.Dense(hidden, cout)
        ...         self.act = ops.Tanh()
        ...
        ...     def construct(self, x):
        ...         x = self.act(self.fc1(x))
        ...         x = self.act(self.fc2(x))
        ...         x = self.fcout(x)
        ...         return x
        >>> model = Net()
        >>> problem = Burgers(model)
        >>> print(problem.pde())
        burgers: u(x, t)Derivative(u(x, t), x) + Derivative(u(x, t), t) - 0.00318309897556901Derivative(u(x, t), (x, 2))
            Item numbers of current derivative formula nodes: 3
        {'burgers': u(x, t)Derivative(u(x, t), x) + Derivative(u(x, t), t) - 0.00318309897556901Derivative(u(x, t),
        (x, 2))}
    """

    def __init__(self, model, loss_fn="mse"):
        self.mu = np.float32(0.01 / np.pi)
        self.x, self.t = symbols('x t')
        self.u = Function('u')(self.x, self.t)
        self.in_vars = [self.x, self.t]
        self.out_vars = [self.u]
        super(Burgers, self).__init__(model, self.in_vars, self.out_vars)
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn

    def pde(self):
        """
        Define Burgers 1-D governing equations based on sympy, abstract method.

        Returns:
            dict, user defined sympy symbolic equations.
        """
        burgers_eq = diff(self.u, (self.t, 1)) + self.u * diff(self.u, (self.x, 1)) - \
            self.mu * diff(self.u, (self.x, 2))

        equations = {"burgers": burgers_eq}
        return equations


class NavierStokes(PDEWithLoss):
    r"""
    2D NavierStokes equation problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): network for training.
        re (float): reynolds number is the ratio of inertia force to viscous force of a fluid. It is a dimensionless
            quantity. Default: ``100.0``.
        loss_fn (Union[str, Cell]): Define the loss function. Default: ``"mse"``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.pde import NavierStokes
        >>> from mindspore import nn, ops
        >>> class Net(nn.Cell):
        ...     def __init__(self, cin=3, cout=3, hidden=10):
        ...         super().__init__()
        ...         self.fc1 = nn.Dense(cin, hidden)
        ...         self.fc2 = nn.Dense(hidden, hidden)
        ...         self.fcout = nn.Dense(hidden, cout)
        ...         self.act = ops.Tanh()
        ...
        ...     def construct(self, x):
        ...         x = self.act(self.fc1(x))
        ...         x = self.act(self.fc2(x))
        ...         x = self.fcout(x)
        ...         return x
        >>> model = Net()
        >>> problem = NavierStokes(model)
        >>> print(problem.pde())
        momentum_x: u(x, y, t)Derivative(u(x, y, t), x) + v(x, y, t)Derivative(u(x, y, t), y) +
        Derivative(p(x, y, t), x) + Derivative(u(x, y, t), t) - 0.00999999977648258Derivative(u(x, y, t), (x, 2)) -
        0.00999999977648258Derivative(u(x, y, t), (y, 2))
            Item numbers of current derivative formula nodes: 6
        momentum_y: u(x, y, t)Derivative(v(x, y, t), x) + v(x, y, t)Derivative(v(x, y, t), y) +
        Derivative(p(x, y, t), y) + Derivative(v(x, y, t), t) - 0.00999999977648258Derivative(v(x, y, t), (x, 2)) -
        0.00999999977648258Derivative(v(x, y, t), (y, 2))
            Item numbers of current derivative formula nodes: 6
        continuty: Derivative(u(x, y, t), x) + Derivative(v(x, y, t), y)
            Item numbers of current derivative formula nodes: 2
        {'momentum_x': u(x, y, t)Derivative(u(x, y, t), x) + v(x, y, t)Derivative(u(x, y, t), y) +
        Derivative(p(x, y, t), x) + Derivative(u(x, y, t), t) - 0.00999999977648258Derivative(u(x, y, t), (x, 2)) -
        0.00999999977648258Derivative(u(x, y, t), (y, 2)),
        'momentum_y': u(x, y, t)Derivative(v(x, y, t), x) + v(x, y, t)Derivative(v(x, y, t), y) +
        Derivative(p(x, y, t), y) + Derivative(v(x, y, t), t) - 0.00999999977648258Derivative(v(x, y, t), (x, 2)) -
        0.00999999977648258Derivative(v(x, y, t), (y, 2)),
        'continuty': Derivative(u(x, y, t), x) + Derivative(v(x, y, t), y)}
    """

    def __init__(self, model, re=100.0, loss_fn="mse"):
        self.number = np.float32(1.0 / re)
        self.x, self.y, self.t = symbols('x y t')
        self.u = Function('u')(self.x, self.y, self.t)
        self.v = Function('v')(self.x, self.y, self.t)
        self.p = Function('p')(self.x, self.y, self.t)
        self.in_vars = [self.x, self.y, self.t]
        self.out_vars = [self.u, self.v, self.p]
        super(NavierStokes, self).__init__(model, self.in_vars, self.out_vars)
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn

    def pde(self):
        """
        Define governing equations based on sympy, abstract method.

        Returns:
            dict, user defined sympy symbolic equations.
        """
        momentum_x = self.u.diff(self.t) + self.u * self.u.diff(self.x) + self.v * self.u.diff(self.y) + \
            self.p.diff(self.x) - self.number * \
            (diff(self.u, (self.x, 2)) + diff(self.u, (self.y, 2)))
        momentum_y = self.v.diff(self.t) + self.u * self.v.diff(self.x) + self.v * self.v.diff(self.y) + \
            self.p.diff(self.y) - self.number * \
            (diff(self.v, (self.x, 2)) + diff(self.v, (self.y, 2)))
        continuty = self.u.diff(self.x) + self.v.diff(self.y)

        equations = {"momentum_x": momentum_x,
                     "momentum_y": momentum_y, "continuty": continuty}
        return equations


class Poisson(PDEWithLoss):
    r"""
    Base class for Poisson 2-D problem based on PDEWithLoss.

    Args:
        model (mindspore.nn.Cell): network for training.
        loss_fn (Union[str, Cell]): Define the loss function. Default: ``"mse"``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindflow.pde import Poisson
        >>> from mindspore import nn, ops
        >>> class Net(nn.Cell):
        ...     def __init__(self, cin=2, cout=1, hidden=10):
        ...         super().__init__()
        ...         self.fc1 = nn.Dense(cin, hidden)
        ...         self.fc2 = nn.Dense(hidden, hidden)
        ...         self.fcout = nn.Dense(hidden, cout)
        ...         self.act = ops.Tanh()
        ...
        ...     def construct(self, x):
        ...         x = self.act(self.fc1(x))
        ...         x = self.act(self.fc2(x))
        ...         x = self.fcout(x)
        ...         return x
        >>> model = Net()
        >>> problem = Poisson(model)
        >>> print(problem.pde())
        poisson: Derivative(u(x, y), (x, 2)) + Derivative(u(x, y), (y, 2)) + 1.0
            Item numbers of current derivative formula nodes: 3
        {'poisson': Derivative(u(x, y), (x, 2)) + Derivative(u(x, y), (y, 2)) + 1.0}
    """

    def __init__(self, model, loss_fn="mse"):
        self.x = Symbol('x')
        self.y = Symbol('y')
        self.normal = Symbol('n')
        self.u = Function('u')(self.x, self.y)

        self.in_vars = [self.x, self.y]
        self.out_vars = [self.u]
        super(Poisson, self).__init__(model, self.in_vars, self.out_vars)
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_metric(loss_fn)
        else:
            self.loss_fn = loss_fn

    def pde(self):
        """
        Define Poisson 2-D governing equations based on sympy, abstract method.

        Returns:
            dict, user defined sympy symbolic equations.
        """
        poisson = diff(self.u, (self.x, 2)) + diff(self.u, (self.y, 2)) + 1.0

        equations = {"poisson": poisson}
        return equations

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
"""Burgers1D"""
import numpy as np
import sympy

from mindspore import Tensor
from mindspore import dtype as mstype

from .pde_with_loss import Burgers
from .parse_sympy import sympy_to_mindspore


class Burgers1D(Burgers):
    r"""
    Burgers 1-D problem based on PDEWithLoss

    Args:
        model (Cell): network for training.
        loss_fn (str): Define the loss function. Default: mse.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self, model, loss_fn="mse"):
        super(Burgers1D, self).__init__(model, loss_fn=loss_fn)
        self.ic_nodes = sympy_to_mindspore(
            self.ic(), self.in_vars, self.out_vars)
        self.bc_nodes = sympy_to_mindspore(
            self.bc(), self.in_vars, self.out_vars)

    def ic(self):
        """
        define initial condition equations based on sympy, abstract method.
        """
        ic_eq = self.u + sympy.sin(np.pi * self.x)
        equations = {"ic": ic_eq}
        return equations

    def bc(self):
        """
        define boundary condition equations based on sympy, abstract method.
        """
        bc_eq = self.u
        equations = {"bc": bc_eq}
        return equations

    def get_loss(self, pde_data, ic_data, bc_data):
        """
        Compute loss of 3 parts: governing equation, initial condition and boundary conditions.

        Args:
            pde_data (Tensor): the input data of governing equations.
            ic_data (Tensor): the input data of initial condition.
            bc_data (Tensor): the input data of boundary condition.
        """
        pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
        pde_loss = self.loss_fn(pde_res[0], Tensor(
            np.array([0.0]), mstype.float32))

        ic_res = self.parse_node(self.ic_nodes, inputs=ic_data)
        ic_loss = self.loss_fn(ic_res[0], Tensor(
            np.array([0.0]), mstype.float32))

        bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
        bc_loss = self.loss_fn(bc_res[0], Tensor(
            np.array([0.0]), mstype.float32))

        return pde_loss + ic_loss + bc_loss

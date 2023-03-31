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
"""Lagrange"""
import sys
from typing import NamedTuple
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Parameter, Tensor, numpy as mnp
from mindspore.nn import Cell
from mindspore.scipy.utils import _to_scalar


class _LagrangeResults(NamedTuple):
    """Object holding optimization results.

    Args:
        succeeded (bool): return true if the solution satisfies the all constrains with respect to save_tol.
        best_value (Tensor): best solution, has the same shape with x0.
        object_value (float): the value of objective function at best_value.
        loss_value (Tensor): the value of constrain functions at best_value.
    """
    succeeded: bool
    best_value: float
    object_value: float
    loss_value: Tensor


class AugmentLagrangeMethod(Cell):
    """minimize lagrange"""

    def __init__(self, objective_func, param, lower, upper, obj_weight, cons_num, constraints, coincide_fun, bound_val):
        """Initialize Minimize lagrange."""
        super(AugmentLagrangeMethod, self).__init__()
        self.objective_func = objective_func
        self.cons_num = cons_num
        self.sigmoid = ops.Sigmoid()
        self.relu = nn.ReLU()
        self.pow = ops.Pow()
        self.bound_val = bound_val
        self.pow_rate = Tensor(2.0)
        self.pow_rate_1 = Tensor(0.5)
        if lower is not None:
            self.lb = Tensor(lower)
        if upper is not None:
            self.ub = Tensor(upper)
        self.param = Parameter(param)
        self.obj_weight = Tensor(obj_weight)
        self.sigma_k = Parameter(Tensor(1.0), requires_grad=False)
        self.alpha = Tensor(0.1)
        self.beta = Tensor(0.9)
        self.epsilon_k = Parameter(1.0 / self.pow(self.sigma_k, self.alpha), requires_grad=False)
        self.eta_k = Parameter(1.0 / self.sigma_k, requires_grad=False)
        self.rho = Tensor(1.1)
        self.constrain = constraints
        self.mul_k = Parameter(mnp.ones((cons_num,), dtype=ms.float32), requires_grad=False)
        self.loss_list = Parameter(mnp.ones((cons_num,), dtype=ms.float32), requires_grad=False)
        self.object_value = Parameter(Tensor(1.0), requires_grad=False)
        self.best_value = Parameter(param, requires_grad=False)
        self.coincide_fun = coincide_fun


    def normalize(self, x):
        """normalize the input_x."""
        if self.bound_val == 0:
            return x
        if self.bound_val == 1:
            return (-x) * x + self.ub
        if self.bound_val == 2:
            return x * x + self.lb
        return (self.ub - self.lb) * self.sigmoid(x) + self.lb


    def get_loss(self, x):
        """get the loss of function."""
        x = self.normalize(x)
        pernalty_func = 0
        if self.coincide_fun is not None:
            tmp_res = self.coincide_fun(x)
            for i in range(self.cons_num):
                cons = self.constrain[i]
                self.loss_list[i] = cons(x, tmp_res)
                pernalty_tmp = self.pow(self.relu(self.mul_k[i] / self.sigma_k + self.loss_list[i]), self.pow_rate)\
                    - self.pow((self.mul_k[i] / self.sigma_k), self.pow_rate)
                pernalty_func += pernalty_tmp
            objective_val = self.objective_func(x, tmp_res)
        else:
            for i in range(self.cons_num):
                cons = self.constrain[i]
                self.loss_list[i] = cons(x)
                pernalty_tmp = self.pow(self.relu(self.mul_k[i] / self.sigma_k + self.loss_list[i]), self.pow_rate)\
                    - self.pow((self.mul_k[i] / self.sigma_k), self.pow_rate)
                pernalty_func += pernalty_tmp
            objective_val = self.objective_func(x)
        loss1 = self.obj_weight * objective_val
        lagrangian_func = loss1 + self.sigma_k / 2 * pernalty_func
        res = [lagrangian_func, self.loss_list, objective_val, x]
        return res


    def update_param(self):
        """update the parameter."""
        v_k = 0
        for i in range(self.cons_num):
            v_k_i = self.pow(max(self.loss_list[i], -self.mul_k[i] / self.sigma_k), self.pow_rate)
            v_k = v_k + v_k_i
        v_k = self.pow(v_k, self.pow_rate_1)
        if v_k <= self.epsilon_k:
            for i in range(self.cons_num):
                self.mul_k[i] = max(self.mul_k[i] + self.sigma_k * self.loss_list[i], 0)
            self.eta_k = self.eta_k / self.sigma_k
            self.epsilon_k = min(self.epsilon_k / (self.sigma_k ** self.beta), sys.float_info.max)
        else:
            self.sigma_k = self.rho * self.sigma_k
            self.eta_k = 1.0 / self.sigma_k
            self.epsilon_k = 1.0 / (self.sigma_k ** self.alpha)


    def construct(self, x):
        res = self.get_loss(self.param)
        lang_func, self.loss_list, self.object_value, self.best_value = res
        x = x * lang_func
        return x


class Loss(Cell):
    """compute loss from data and label"""
    def __init__(self):
        """Initialize the loss."""
        super(Loss, self).__init__()

    def construct(self, data, label):
        diff = data - label
        return diff


def check_tol(f0, object_value, save_tol, loss_list):
    """check tolerance."""
    chk_res = False
    if (np.array(loss_list) - save_tol).max() < 0 and object_value < f0:
        chk_res = True
    return chk_res


def minimize_lagrange(func, x0, constraints, save_tol=None, obj_weight=1.0, lower=None, upper=None,
                      learning_rate=0.1, coincide_fun=None, rounds=10, steps=1000, log_sw=False):
    """minimize lagrange."""
    cons_num = len(constraints)
    if lower is None and upper is None:
        bound_val = 0
    elif lower is None and upper is not None:
        bound_val = 1
    elif lower is not None and upper is None:
        bound_val = 2
    else:
        bound_val = 3

    net = AugmentLagrangeMethod(func, x0, lower, upper, obj_weight, cons_num, constraints, coincide_fun, bound_val)
    loss = Loss()
    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=learning_rate, use_locking=True,
                                                use_nesterov=False)
    loss_net = nn.WithLossCell(net, loss)
    label = Tensor(0.0, ms.float32)
    image = Tensor(1.0, ms.float32)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)
    f0 = 0
    best_value = x0.asnumpy()
    chk_res = False
    loss_list = mnp.ones((cons_num,), dtype=ms.float32)
    if save_tol is None:
        save_tol = mnp.ones((cons_num,), dtype=ms.float32)
    for j in range(rounds):
        for i in range(steps):
            lang_val = train_net(image, label)
            if log_sw:
                print(F"round:{j}, step:{i}")
                print(F"object_value:{net.object_value.value()}, loss_value:{net.loss_list.value()}, \
                            lagrange_value:{lang_val}")
            if i == 0 and j == 0:
                f0 = net.object_value.value().asnumpy()
            chkres = check_tol(f0, net.object_value.value().asnumpy(), save_tol, net.loss_list.value().asnumpy())
            if chkres:
                chk_res = True
                loss_list = net.loss_list.value().asnumpy()
                best_value = net.best_value.value().asnumpy()
                f0 = net.object_value.value().asnumpy()
        net.update_param()
    res = _LagrangeResults(succeeded=_to_scalar(chk_res), best_value=best_value, object_value=f0, loss_value=loss_list)
    return res

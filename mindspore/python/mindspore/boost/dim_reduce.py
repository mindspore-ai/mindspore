# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""dim_reduce"""
from __future__ import absolute_import

import math
import numpy as np
from mindspore.nn.cell import Cell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common import dtype as mstype


__all__ = ["DimReduce"]


_scale_grad = C.MultitypeFuncGraph("_scale_grad")


@_scale_grad.register("Tensor", "Tensor")
def _scale_grad_process(scale, grad):
    grad = F.cast(grad, mstype.float32)
    grad = P.Div()(grad, scale)
    return grad


_save_weight = C.MultitypeFuncGraph("_save_weight")


@_save_weight.register("Tensor", "Tensor")
def _save_weight_process(parameter, new_parameter):
    P.Assign()(parameter, new_parameter)
    return parameter


_pca_projection = C.MultitypeFuncGraph("_pca_projection")


@_pca_projection.register("Tensor", "Tensor")
def _pca_projection_process(pca_mat, grad):
    grad_k = P.MatMul()(pca_mat, F.reshape(grad, (-1, 1)))
    return grad_k


_pca_back_projection = C.MultitypeFuncGraph("_pca_back_projection")


@_pca_back_projection.register("Tensor", "Tensor", "Tensor")
def _pca_back_projection_process(grad_k, pca_mat, grad):
    grad_proj = P.MatMul()(F.transpose(pca_mat, (1, 0)), grad_k)
    grad_proj_reshape = F.reshape(grad_proj, F.shape(grad))
    return grad_proj_reshape


_update_grad_res_momentum = C.MultitypeFuncGraph("_update_grad_res_momentum")


@_update_grad_res_momentum.register("Float32", "Float32", "Tensor", "Tensor", "Tensor")
def _update_grad_res_momentum_process(gamma, alpha, grad_res_momentum, grad, grad_proj):
    grad_res_momentum_new = gamma * grad_res_momentum + grad - grad_proj
    P.Assign()(grad_res_momentum, grad_res_momentum_new)
    res = alpha * grad_res_momentum_new
    return res


_get_delta_weight = C.MultitypeFuncGraph("_get_delta_weight")


@_get_delta_weight.register("Tensor", "Tensor", "Tensor")
def _get_delta_weight_process(rho, dn, grad_res_momentum):
    delta_weight = grad_res_momentum - rho * dn
    return delta_weight


class DimReduce(Cell):
    r"""
    The dimension reduce training, is a novel algorithm for accelerating convergence of Deep Learning models.

    .. math::

            \begin{align}
            grad\_k &= pca\_mat \cdot grad\\
            dk &= - bk \cdot grad\_k\\
            sk &= rho ^ m \cdot dk\\
            delta\_loss &= sigma \cdot grad\_k.T \cdot sk
            \end{align}

    Here:

    - pca_mat (array): Shape (k*n), k is part of n_components, n is the size of weight.
    - bk (array): Shape (k*k), is the symmetric positive definite matrix in Quasi-Newton method.

    we need to find the m satisfy:

    .. math::
            new\_loss < old\_loss + delta\_loss

    Then, get delta_grad to update the weights for model:

    .. math::

            \begin{align}
            grad\_k\_proj &= pca\_mat.T \cdot grad\_k\\
            new\_grad\_momentum &= gamma \cdot old\_grad\_momentum + grad - grad\_k\_proj\\
            delta\_grad &= alpha \cdot new\_grad\_momentum - pca\_mat.T \cdot sk
            \end{align}

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Union[Cell]): Optimizer for updating the weights.
        weight (Tuple(Parameter)): Tuple of parameters.
        pca_mat_local (numpy.ndarray): For PCA operation, k*n, k is part of n_components, n is the size of weight.
        n_components (int): PCA.components.
        rho (float): Coefficient.
        gamma (float): Coefficient.
        alpha (float): Coefficient.
        sigma (float): Coefficient.
        rank (int): Rank number.
        rank_size (int): Rank size.

    Inputs:
        - **loss** (Tensor) - Tensor with shape :math:`()`.
        - **old_grad** (Tuple(Tensor)) - Tuple of gradient tensors.
        - **weight** (Tuple(Tensor)) - Tuple of parameters.
        - **weight_clone** (Tuple(Tensor)) - clone of weight
        - **\*inputs** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        - **loss** (Tensor) - Tensor with shape :math:`()`.
    """
    def __init__(self, network, optimizer, weight, pca_mat_local, n_components, rho, gamma, alpha, sigma, rank,
                 rank_size):
        super(DimReduce, self).__init__()
        self.network = network
        self.optimizer = optimizer
        self.rank = rank
        self.rank_size = rank_size
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma

        self.float_type = mstype.float32
        self._set_rho_list(rho)
        self._set_local_pca_mat(pca_mat_local, n_components, weight)
        self._set_init_parameter(weight)

        self.hyper_map = C.HyperMap()
        self.concat = P.Concat()
        self.matmul = P.MatMul()
        self.mul = P.Mul()
        self.add = P.Add()

    def construct(self, loss, old_grad, loss_scale, weight, weight_clone, *inputs):
        gk, old_loss, gk_local = self._generate_gk(weight, loss, old_grad, loss_scale)

        _save_weight(self.gk_last_back, self.gk_last)
        _save_weight(self.bk_back, self.bk)

        dk = self._apply_quasi_newton_update(gk)
        if self.dk_pad_flag:
            dk_pad = self.concat((dk, self.dk_pad_part))
        else:
            dk_pad = dk
        dk_local = dk_pad[self.start_index: self.end_index, :]

        dn_local = self.hyper_map(F.partial(_pca_back_projection, dk_local), self.pca_list_local, old_grad)
        grad_proj_local = self.hyper_map(F.partial(_pca_back_projection, gk_local), self.pca_list_local, old_grad)
        dn = self.dn_init if self.rank_size > 1 else dn_local
        grad_proj = self.grad_proj_init if self.rank_size > 1 else grad_proj_local
        if self.rank_size > 1:
            for broadcast in self.broadcast_list:
                dn_part = broadcast(dn_local)
                dn = self.hyper_map(self.add, dn, dn_part)
                grad_proj_part = broadcast(grad_proj_local)
                grad_proj = self.hyper_map(self.add, grad_proj, grad_proj_part)

        rho, find = self._line_search(gk, dk, dn, old_loss, weight, weight_clone, *inputs)
        if not find:
            _save_weight(self.gk_last, self.gk_last_back)
            _save_weight(self.bk, self.bk_back)

        clone = self._res_loss(old_grad, grad_proj, weight, weight_clone, rho, dn)
        return F.depend(loss, clone)

    def _set_rho_list(self, rho):
        """set rho list info."""
        self.max_search_time = 2
        self.rho_list = []
        for i in range(self.max_search_time):
            self.rho_list.append(Tensor(np.power(rho, i), dtype=self.float_type))
        self.rho_list.append(Tensor(0, dtype=self.float_type))

    def _set_local_pca_mat(self, pca_mat_local, n_components, parameter_tuple):
        """set pca info."""
        self.n_components = n_components
        local_dim = math.ceil(self.n_components // self.rank_size)

        self.start_index = self.rank * local_dim
        self.end_index = (self.rank + 1) * local_dim

        start = 0
        self.pca_list_local = ()
        for param in parameter_tuple:
            size = np.shape(param.asnumpy().reshape((-1, 1)))[0]
            self.pca_list_local += (Tensor(pca_mat_local[:, start:start + size], dtype=self.float_type),)
            start += size

        self.dk_pad_flag = False
        pad_num = self.rank_size * local_dim - self.n_components
        if pad_num:
            self.dk_pad_flag = True
            self.dk_pad_part = Tensor(np.zeros([pad_num, 1]), dtype=self.float_type)

        if self.rank_size > 1:
            self.broadcast_list = []
            for i in range(self.rank_size):
                broadcast = P.Broadcast(i)
                self.broadcast_list.append(broadcast)
            self.allreduce = P.AllReduce()
            self.allgather = P.AllGather()

    def _set_init_parameter(self, parameter_tuple):
        """init parameters."""
        self.true_flag = Tensor(True)
        self.false_flag = Tensor(False)
        self.epsilon = np.power(10.0, -20)
        self.gk_last = Parameter(Tensor(np.zeros([self.n_components, 1]), dtype=self.float_type), name="gk_last")
        self.gk_last_init = Parameter(Tensor(False), name="gk_last_init")
        self.bk = Parameter(Tensor(np.eye(self.n_components), dtype=self.float_type), name="bk")
        self.sk = Parameter(Tensor(np.zeros([self.n_components, 1]), dtype=self.float_type), name="sk")
        self.eye = Tensor(np.eye(self.n_components), dtype=self.float_type)
        self.grad_res_momentum = ParameterTuple(parameter_tuple).clone(prefix="grad_res_momentum", init="zeros")
        self.gk_last_back = Parameter(Tensor(np.zeros([self.n_components, 1]), dtype=self.float_type),
                                      name="gk_last_back")
        self.bk_back = Parameter(Tensor(np.eye(self.n_components), dtype=self.float_type), name="bk_back")
        self.grad_proj_init = ParameterTuple(parameter_tuple).clone(prefix="grad_proj_init", init="zeros")
        self.dn_init = ParameterTuple(parameter_tuple).clone(prefix="dn_init", init="zeros")

    def _res_loss(self, old_grad, grad_proj, weight, weight_clone, rho, dn):
        """update loss"""
        update_grad = self.hyper_map(F.partial(_update_grad_res_momentum, self.gamma, self.alpha),
                                     self.grad_res_momentum, old_grad, grad_proj)
        delta_weight = self.hyper_map(F.partial(_get_delta_weight, rho), dn, update_grad)
        update = self.optimizer(delta_weight)
        weight = F.depend(weight, update)
        clone = self.hyper_map(_save_weight, weight_clone, weight)
        return clone

    def _generate_gk(self, weight, loss, old_grad, loss_scale):
        """generate gk"""
        weight = F.depend(weight, loss)
        old_grad = F.depend(old_grad, weight)
        old_grad = self.hyper_map(F.partial(_scale_grad, loss_scale), old_grad)
        old_loss = self.allreduce(loss) // self.rank_size if self.rank_size > 1 else loss

        gk_local = self.hyper_map(_pca_projection, self.pca_list_local, old_grad)
        gk_local = F.addn(gk_local)
        gk_pad = self.allgather(gk_local) if self.rank_size > 1 else gk_local
        gk_pad = F.reshape(gk_pad, (-1, 1))
        gk = gk_pad[0:self.n_components, :]
        return gk, old_loss, gk_local

    def _line_search(self, gk, dk, dn, old_loss, weight, weight_clone, *inputs):
        """line search rho."""
        res = self.rho_list[-1]
        find = self.false_flag
        for i in range(self.max_search_time):
            find = self._find_rho(gk, dk, dn, old_loss, weight, weight_clone, self.rho_list[i], *inputs)
            if find:
                res = self.rho_list[i]
                break
        return res, find

    def _find_rho(self, gk, dk, dn, old_loss, weight, weight_clone, rho, *inputs):
        """search rho."""
        res = self.false_flag
        sn = self.hyper_map(F.partial(self.mul, -1 * rho), dn)
        sn = F.depend(sn, old_loss)
        update = self.optimizer(sn)
        new_loss = F.depend(self.network(*inputs), update)
        if self.rank_size > 1:
            new_loss = self.allreduce(new_loss) // self.rank_size
        old_loss_delta = old_loss + self.sigma * rho * F.squeeze(self.matmul(F.transpose(gk, (1, 0)), dk))
        if old_loss_delta > new_loss:
            _save_weight(self.sk, rho * dk)
            res = self.true_flag
        weight_clone = F.depend(weight_clone, old_loss_delta)
        restore = self.hyper_map(_save_weight, weight, weight_clone)
        res = F.depend(res, restore)
        return res

    def _apply_quasi_newton_update(self, gk):
        """apply quasi_newton update."""
        if self.gk_last_init:
            yk = gk - self.gk_last
            g = self.matmul(F.transpose(yk, (1, 0)), self.sk)
            g = F.squeeze(g)
            if g > self.epsilon:
                pk = 1. / g
                t1 = self.eye - self.matmul(pk * yk, F.transpose(self.sk, (1, 0)))
                new_bk = self.matmul(self.matmul(F.transpose(t1, (1, 0)), self.bk), t1) + \
                         self.matmul(pk * self.sk, F.transpose(self.sk, (1, 0)))
                _save_weight(self.bk, new_bk)
        else:
            _save_weight(self.gk_last_init, self.true_flag)
        _save_weight(self.gk_last, gk)
        dk = -1 * self.matmul(self.bk, gk)
        return dk

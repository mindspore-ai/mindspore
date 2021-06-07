# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Define the PINNs network for the Navier-Stokes equation."""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import TruncatedNormal, Zero, initializer
from mindspore.ops import constexpr


@constexpr
def _generate_ones(batch_size):
    arr = np.ones((batch_size, 1), np.float32)
    return Tensor(arr, mstype.float32)


@constexpr
def _generate_zeros(batch_size):
    arr = np.zeros((batch_size, 1), np.float32)
    return Tensor(arr, mstype.float32)


class neural_net(nn.Cell):
    """
    Neural net to fit the wave function

    Args:
        layers (list(int)): num of neurons for each layer
        lb (np.array): lower bound (x, t) of domain
        ub (np.array): upper bound (x, t) of domain
    """
    def __init__(self, layers, lb, ub):
        super(neural_net, self).__init__()
        self.layers = layers
        self.concat = ops.Concat(axis=1)
        self.lb = Tensor(lb, mstype.float32)
        self.ub = Tensor(ub, mstype.float32)

        self.tanh = ops.Tanh()
        self.add = ops.Add()
        self.matmul = ops.MatMul()

        self.w0 = self._init_weight_xavier(0)
        self.b0 = self._init_biase(0)
        self.w1 = self._init_weight_xavier(1)
        self.b1 = self._init_biase(1)
        self.w2 = self._init_weight_xavier(2)
        self.b2 = self._init_biase(2)
        self.w3 = self._init_weight_xavier(3)
        self.b3 = self._init_biase(3)
        self.w4 = self._init_weight_xavier(4)
        self.b4 = self._init_biase(4)
        self.w5 = self._init_weight_xavier(5)
        self.b5 = self._init_biase(5)
        self.w6 = self._init_weight_xavier(6)
        self.b6 = self._init_biase(6)
        self.w7 = self._init_weight_xavier(7)
        self.b7 = self._init_biase(7)
        self.w8 = self._init_weight_xavier(8)
        self.b8 = self._init_biase(8)

    def construct(self, x, y, t):
        """Forward propagation"""
        X = self.concat((x, y, t))
        X = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0

        X = self.tanh(self.add(self.matmul(X, self.w0), self.b0))
        X = self.tanh(self.add(self.matmul(X, self.w1), self.b1))
        X = self.tanh(self.add(self.matmul(X, self.w2), self.b2))
        X = self.tanh(self.add(self.matmul(X, self.w3), self.b3))
        X = self.tanh(self.add(self.matmul(X, self.w4), self.b4))
        X = self.tanh(self.add(self.matmul(X, self.w5), self.b5))
        X = self.tanh(self.add(self.matmul(X, self.w6), self.b6))
        X = self.tanh(self.add(self.matmul(X, self.w7), self.b7))
        X = self.add(self.matmul(X, self.w8), self.b8)

        return X[:, 0:1], X[:, 1:2]

    def _init_weight_xavier(self, layer):
        """
        Initialize weight for the ith layer
        """
        in_dim = self.layers[layer]
        out_dim = self.layers[layer+1]
        std = np.sqrt(2/(in_dim + out_dim))
        name = 'w' + str(layer)
        return Parameter(default_input=initializer(TruncatedNormal(std), [in_dim, out_dim], mstype.float32),
                         name=name, requires_grad=True)

    def _init_biase(self, layer):
        """
        Initialize biase for the ith layer
        """
        name = 'b' + str(layer)
        return Parameter(default_input=initializer(Zero(), self.layers[layer+1], mstype.float32),
                         name=name, requires_grad=True)


class Grad_2_1(nn.Cell):
    """
    Net has 3 inputs and 2 outputs. Using the first output to compute gradient.
    """
    def __init__(self, net):
        super(Grad_2_1, self).__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, sens_param=True)

    def construct(self, x, y, t):
        sens_1 = _generate_ones(x.shape[0])
        sens_2 = _generate_zeros(x.shape[0])
        return self.grad(self.net)(x, y, t, (sens_1, sens_2))


class Grad_2_2(nn.Cell):
    """
    Net has 3 inputs and 2 outputs. Using the third output to compute gradient.
    """
    def __init__(self, net):
        super(Grad_2_2, self).__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, sens_param=True)

    def construct(self, x, y, t):
        sens_1 = _generate_zeros(x.shape[0])
        sens_2 = _generate_ones(x.shape[0])
        return self.grad(self.net)(x, y, t, (sens_1, sens_2))


class Grad_3_1(nn.Cell):
    """
    Net has 3 inputs and 3 outputs. Using the first output to compute gradient.
    """
    def __init__(self, net):
        super(Grad_3_1, self).__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.gradop = self.grad(self.net)

    def construct(self, x, y, t):
        sens_1 = _generate_ones(x.shape[0])
        sens_2 = _generate_zeros(x.shape[0])
        sens_3 = _generate_zeros(x.shape[0])
        return self.grad(self.net)(x, y, t, (sens_1, sens_2, sens_3))


class Grad_3_2(nn.Cell):
    """
    Net has 3 inputs and 3 outputs. Using the second output to compute gradient.
    """
    def __init__(self, net):
        super(Grad_3_2, self).__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, sens_param=True)

    def construct(self, x, y, t):
        sens_1 = _generate_zeros(x.shape[0])
        sens_2 = _generate_ones(x.shape[0])
        sens_3 = _generate_zeros(x.shape[0])
        return self.grad(self.net)(x, y, t, (sens_1, sens_2, sens_3))


class PINNs_navier(nn.Cell):
    """
    PINNs for the Navier-Stokes equation.
    """
    def __init__(self, layers, lb, ub):
        super(PINNs_navier, self).__init__()
        self.lambda1 = Parameter(default_input=initializer(Zero(), 1, mstype.float32),
                                 name='lambda1', requires_grad=True)
        self.lambda2 = Parameter(default_input=initializer(Zero(), 1, mstype.float32),
                                 name='lambda2', requires_grad=True)

        self.mul = ops.Mul()
        self.add = ops.Add()
        self.nn = neural_net(layers, lb, ub)

        # first order gradient
        self.dpsi = Grad_2_1(self.nn)
        self.dpsi_dv = Grad_2_1(self.nn)
        self.dpsi_duy = Grad_2_1(self.nn)
        self.dpsi_dvy = Grad_2_1(self.nn)

        self.dp = Grad_2_2(self.nn)

        # second order gradient
        self.du = Grad_3_2(self.dpsi)
        self.du_duy = Grad_3_2(self.dpsi_duy)

        self.dv = Grad_3_1(self.dpsi_dv)
        self.dv_dvy = Grad_3_1(self.dpsi_dvy)

        # third order gradient
        self.dux = Grad_3_1(self.du)
        self.duy = Grad_3_2(self.du_duy)
        self.dvx = Grad_3_1(self.dv)
        self.dvy = Grad_3_2(self.dv_dvy)

    def construct(self, X):
        """forward propagation"""
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]

        ans_nn = self.nn(x, y, t)
        p = ans_nn[1]

        # first order gradient
        d_psi = self.dpsi(x, y, t)
        v = -d_psi[0]
        u = d_psi[1]

        d_p = self.dp(x, y, t)
        px = d_p[0]
        py = d_p[1]

        # second order gradient
        d_u = self.du(x, y, t)
        ux = d_u[0]
        uy = d_u[1]
        ut = d_u[2]

        d_v = self.dv(x, y, t)
        vx = -d_v[0]
        vy = -d_v[1]
        vt = -d_v[2]

        # third order gradient
        d_ux = self.dux(x, y, t)
        uxx = d_ux[0]
        d_uy = self.duy(x, y, t)
        uyy = d_uy[1]

        d_vx = self.dvx(x, y, t)
        vxx = -d_vx[0]
        d_vy = self.dvy(x, y, t)
        vyy = -d_vy[1]

        # regularizer of the PDE (Navier-Stokes)
        fu1 = self.add(self.mul(u, ux), self.mul(v, uy))
        fu1 = self.mul(self.lambda1, fu1)
        fu2 = self.add(uxx, uyy)
        fu2 = self.mul(self.lambda2, uyy)
        fu2 = self.mul(fu2, -1.0)
        fu = ut + fu1 + px + fu2

        fv1 = self.add(self.mul(u, vx), self.mul(v, vy))
        fv1 = self.mul(self.lambda1, fv1)
        fv2 = self.add(vxx, vyy)
        fv2 = self.mul(self.lambda2, fv2)
        fv2 = self.mul(fv2, -1.0)
        fv = vt + fv1 + py + fv2

        return u, v, p, fu, fv
    
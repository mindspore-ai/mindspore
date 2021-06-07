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
"""Define the PINNs network for the Schrodinger equation."""
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

    def construct(self, x, t):
        """Forward propagation"""
        X = self.concat((x, t))
        X = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0

        X = self.tanh(self.add(self.matmul(X, self.w0), self.b0))
        X = self.tanh(self.add(self.matmul(X, self.w1), self.b1))
        X = self.tanh(self.add(self.matmul(X, self.w2), self.b2))
        X = self.tanh(self.add(self.matmul(X, self.w3), self.b3))
        X = self.add(self.matmul(X, self.w4), self.b4)

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


class Grad_1(nn.Cell):
    """
    Net has 2 inputs and 2 outputs. Using the first output to compute gradient.
    """
    def __init__(self, net):
        super(Grad_1, self).__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, sens_param=True)

    def construct(self, x, t):
        sens_1 = _generate_ones(x.shape[0])
        sens_2 = _generate_zeros(x.shape[0])
        return self.grad(self.net)(x, t, (sens_1, sens_2))


class Grad_2(nn.Cell):
    """
    Net has 2 inputs and 2 outputs. Using the second output to compute gradient.
    """
    def __init__(self, net):
        super(Grad_2, self).__init__()
        self.net = net
        self.grad = ops.GradOperation(get_all=True, sens_param=True)

    def construct(self, x, t):
        sens_1 = _generate_zeros(x.shape[0])
        sens_2 = _generate_ones(x.shape[0])
        return self.grad(self.net)(x, t, (sens_1, sens_2))


class PINNs(nn.Cell):
    """
    PINNs for the Schrodinger equation.
    """
    def __init__(self, layers, lb, ub):
        super(PINNs, self).__init__()
        self.nn = neural_net(layers, lb, ub)
        self.du = Grad_1(self.nn)
        self.dv = Grad_2(self.nn)
        self.dux = Grad_1(self.du)
        self.dvx = Grad_1(self.dv)

        self.add = ops.Add()
        self.pow = ops.Pow()
        self.mul = ops.Mul()

    def construct(self, X):
        """forward propagation"""
        x = X[:, 0:1]
        t = X[:, 1:2]
        u, v = self.nn(x, t)
        ux, ut = self.du(x, t)
        vx, vt = self.dv(x, t)
        uxx, _ = self.dux(x, t)
        vxx, _ = self.dvx(x, t)

        square_sum = self.add(self.pow(u, 2), self.pow(v, 2))

        fu1 = self.mul(vxx, 0.5)
        fu2 = self.mul(square_sum, v)
        fu = self.add(self.add(ut, fu1), fu2)

        fv1 = self.mul(uxx, -0.5)
        fv2 = self.mul(square_sum, u)
        fv2 = self.mul(fv2, -1.0)
        fv = self.add(self.add(vt, fv1), fv2)

        return u, v, ux, vx, fu, fv

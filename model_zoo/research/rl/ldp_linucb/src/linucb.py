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
"""
Linear UCB with locally differentially private.
"""

import math
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class LinUCB(nn.Cell):
    """
    Linear UCB with locally differentially private bandits learning.

    Args:
        context_dim(int): dim of input feature.
        epsilon(float): epsilon for private parameter.
        delta(float): delta for private parameter.
        alpha(float): failure probability.
        T(float/int): number of iterations.

    Returns:
        Tuple of Tensors: gradients to update parameters and optimal action.
    """
    def __init__(self, context_dim, epsilon=100, delta=0.1, alpha=0.1, T=1e5):
        super(LinUCB, self).__init__()
        self.matmul = P.MatMul()
        self.expand_dims = P.ExpandDims()
        self.transpose = P.Transpose()
        self.reduce_sum = P.ReduceSum()
        self.squeeze = P.Squeeze(1)
        self.argmax = P.Argmax()
        self.reduce_max = P.ReduceMax()

        # Basic variables
        self._context_dim = context_dim
        self._epsilon = epsilon
        self._delta = delta
        self._alpha = alpha
        self._T = int(T)

        # Parameters
        self._V = Tensor(
            np.zeros(
                (context_dim,
                 context_dim),
                dtype=np.float32))
        self._u = Tensor(np.zeros((context_dim,), dtype=np.float32))
        self._theta = Tensor(np.zeros((context_dim,), dtype=np.float32))

        # \sigma = 4*\sqrt{2*\ln{\farc{1.25}{\delta}}}/\epsilon
        self._sigma = 4 * \
            math.sqrt(math.log(1.25 / self._delta)) / self._epsilon
        self._c = 0.1
        self._step = 1
        self._regret = 0
        self._current_regret = 0
        self.inverse_matrix()

    @property
    def theta(self):
        return self._theta

    @property
    def regret(self):
        return self._regret

    @property
    def current_regret(self):
        return self._current_regret

    def inverse_matrix(self):
        """compute the inverse matrix of parameter matrix."""
        Vc = self._V + Tensor(np.eye(self._context_dim,
                                     dtype=np.float32)) * self._c
        self._Vc_inv = Tensor(np.linalg.inv(Vc.asnumpy()), mindspore.float32)

    def update_status(self, step):
        """update status variables."""
        t = max(step, 1)
        T = self._T
        d = self._context_dim
        alpha = self._alpha
        sigma = self._sigma

        gamma = sigma * \
            math.sqrt(t) * (4 * math.sqrt(d) + 2 * math.log(2 * T / alpha))
        self._c = 2 * gamma
        self._beta = 2 * sigma * math.sqrt(d * math.log(T)) + (math.sqrt(
            3 * gamma) + sigma * math.sqrt(d * t / gamma)) * d * math.log(T)

    def construct(self, x, rewards):
        """compute the perturbed gradients for parameters."""
        # Choose optimal action
        x_transpose = self.transpose(x, (1, 0))
        scores_a = self.squeeze(self.matmul(x, self.expand_dims(self._theta, 1)))
        scores_b = x_transpose * self.matmul(self._Vc_inv, x_transpose)
        scores_b = self.reduce_sum(scores_b, 0)
        scores = scores_a + self._beta * scores_b
        max_a = self.argmax(scores)
        xa = x[max_a]
        xaxat = self.matmul(self.expand_dims(xa, -1), self.expand_dims(xa, 0))
        y = rewards[max_a]
        y_max = self.reduce_max(rewards)
        y_diff = y_max - y
        self._current_regret = float(y_diff.asnumpy())
        self._regret += self._current_regret

        # Prepare noise
        B = np.random.normal(0, self._sigma, size=xaxat.shape)
        B = np.triu(B)
        B += B.transpose() - np.diag(B.diagonal())
        B = Tensor(B.astype(np.float32))
        Xi = np.random.normal(0, self._sigma, size=xa.shape)
        Xi = Tensor(Xi.astype(np.float32))

        # Add noise and update parameters
        return xaxat + B, xa * y + Xi, max_a

    def server_update(self, xaxat, xay):
        """update parameters with perturbed gradients."""
        self._V += xaxat
        self._u += xay
        self.inverse_matrix()
        theta = self.matmul(self._Vc_inv, self.expand_dims(self._u, 1))
        self._theta = self.squeeze(theta)

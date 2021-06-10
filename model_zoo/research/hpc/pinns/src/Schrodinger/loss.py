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
"""Loss function for PINNs (Schrodinger)"""
import mindspore.common.dtype as mstype
from mindspore import nn, ops


class PINNs_loss(nn.Cell):
    """
    Loss of the PINNs network (Schrodinger), only works with full-batch training. Training data are organized in
    the following order: initial condition points ([0:n0]), boundary condition points ([n0:(n0+2*nb)]),
    collocation points ([(n0+2*nb)::])
    """
    def __init__(self, n0, nb, nf, reduction='sum'):
        super(PINNs_loss, self).__init__(reduction)
        self.n0 = n0
        self.nb = nb
        self.nf = nf
        self.zeros = ops.Zeros()
        self.mse = nn.MSELoss(reduction='mean')
        self.f_target = self.zeros((self.nf, 1), mstype.float32)

    def construct(self, pred, target):
        """
        pred: prediction value (u, v, ux, vx, fu, fv)
        target: target[:, 0:1] = u_target, target[:, 0:2] = v_target
        """
        u0_pred = pred[0][0:self.n0, 0:1]
        u0 = target[0:self.n0, 0:1]
        v0_pred = pred[1][0:self.n0, 0:1]
        v0 = target[0:self.n0, 1:2]

        u_lb_pred = pred[0][self.n0:(self.n0+self.nb), 0:1]
        u_ub_pred = pred[0][(self.n0+self.nb):(self.n0+2*self.nb), 0:1]
        v_lb_pred = pred[1][self.n0:(self.n0+self.nb), 0:1]
        v_ub_pred = pred[1][(self.n0+self.nb):(self.n0+2*self.nb), 0:1]

        ux_lb_pred = pred[2][self.n0:(self.n0+self.nb), 0:1]
        ux_ub_pred = pred[2][(self.n0+self.nb):(self.n0+2*self.nb), 0:1]
        vx_lb_pred = pred[3][self.n0:(self.n0+self.nb), 0:1]
        vx_ub_pred = pred[3][(self.n0+self.nb):(self.n0+2*self.nb), 0:1]

        fu_pred = pred[4][(self.n0+2*self.nb)::, 0:1]
        fv_pred = pred[5][(self.n0+2*self.nb)::, 0:1]

        mse_u_0 = self.mse(u0_pred, u0)
        mse_v_0 = self.mse(v0_pred, v0)
        mse_u_b = self.mse(u_lb_pred, u_ub_pred)
        mse_v_b = self.mse(v_lb_pred, v_ub_pred)
        mse_ux_b = self.mse(ux_lb_pred, ux_ub_pred)
        mse_vx_b = self.mse(vx_lb_pred, vx_ub_pred)
        mse_fu = self.mse(fu_pred, self.f_target)
        mse_fv = self.mse(fv_pred, self.f_target)

        ans = mse_u_0 + mse_v_0 + mse_u_b + mse_v_b + mse_ux_b + mse_vx_b + mse_fu + mse_fv

        return ans

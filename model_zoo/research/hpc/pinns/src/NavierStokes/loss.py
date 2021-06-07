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
"""Loss function for PINNs (Navier-Stokes)"""
from mindspore import nn


class PINNs_loss_navier(nn.Cell):
    """
    Loss of PINNs (Navier-Stokes). Loss = mse loss + regularizer term from the PDE.
    """
    def __init__(self):
        super(PINNs_loss_navier, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def construct(self, pred, target):
        """
        pred: preditiction of PINNs (Navier-Stokes), pred = (u, v, p, fu, fv)
        target: targeted value of (u, v)
        """
        u_pred = pred[0]
        u_target = target[:, 0:1]
        v_pred = pred[1]
        v_target = target[:, 1:2]
        fu_pred = pred[3]
        fv_pred = pred[4]
        f_target = target[:, 2:3]

        mse_u = self.mse(u_pred, u_target)
        mse_v = self.mse(v_pred, v_target)
        mse_fu = self.mse(fu_pred, f_target)
        mse_fv = self.mse(fv_pred, f_target)

        ans = mse_u + mse_v + mse_fu + mse_fv
        return ans

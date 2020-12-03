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
"""Calculate virial of atoms."""
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.ops import operations as P


class ProdVirialSeA(nn.Cell):
    """calculate virial."""
    def __init__(self):
        super(ProdVirialSeA, self).__init__()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.rsum = P.ReduceSum()
        self.rksum = P.ReduceSum(keep_dims=True)
        self.broadcastto1 = P.BroadcastTo((1, 192, 138, 4, 3, 3))
        self.broadcastto2 = P.BroadcastTo((1, 192, 138, 4, 3))
        self.broadcastto3 = P.BroadcastTo((1, 192, 138, 3))
        self.expdims = P.ExpandDims()

    def construct(self, net_deriv_reshape, descrpt_deriv, rij, nlist):
        """construct function."""
        descrpt_deriv = self.cast(descrpt_deriv, mstype.float32)
        descrpt_deriv = self.reshape(descrpt_deriv, (1, 192, 138, 4, 3))

        net_deriv_reshape = self.cast(net_deriv_reshape, mstype.float32)
        net_deriv_reshape = self.reshape(net_deriv_reshape, (1, 192, 138, 4))
        net_deriv_reshape = self.expdims(net_deriv_reshape, 4)
        net_deriv_reshape = self.broadcastto2(net_deriv_reshape)

        rij = self.cast(rij, mstype.float32)
        rij = self.reshape(rij, (1, 192, 138, 3))
        rij = self.expdims(rij, 3)
        rij = self.expdims(rij, 4)
        rij = self.broadcastto1(rij)

        nlist = self.cast(nlist, mstype.int32)
        nlist = self.reshape(nlist, (1, 192, 138))
        nlist = self.expdims(nlist, 3)
        nlist = self.broadcastto3(nlist)

        tmp = descrpt_deriv * net_deriv_reshape

        b_blist = self.cast(nlist > -1, mstype.int32)
        b_blist = self.expdims(b_blist, 3)
        b_blist = self.broadcastto2(b_blist)

        tmp_1 = tmp * b_blist
        tmp_1 = self.expdims(tmp_1, 5)
        tmp_1 = self.broadcastto1(tmp_1)

        out = tmp_1 * rij
        out = self.rsum(out, (1, 2, 3))
        return out

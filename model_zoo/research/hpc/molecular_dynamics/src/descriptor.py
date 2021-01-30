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
"""The construction of the descriptor."""
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P


class ComputeRij(nn.Cell):
    """compute rij."""
    def __init__(self):
        super(ComputeRij, self).__init__()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.rsum = P.ReduceSum()
        self.broadcastto = P.BroadcastTo((1, 192 * 138))
        self.broadcastto1 = P.BroadcastTo((1, 192, 138, 3))
        self.expdims = P.ExpandDims()
        self.concat = P.Concat(axis=1)
        self.gather = P.Gather()
        self.mul = P.Mul()
        self.slice = P.Slice()

    def construct(self, d_coord_tensor, nlist_tensor, frames):
        """construct function."""
        d_coord_tensor = self.cast(d_coord_tensor, mstype.float32)
        d_coord_tensor = self.reshape(d_coord_tensor, (1, -1, 3))
        coord_tensor = self.slice(d_coord_tensor, (0, 0, 0), (1, 192, 3))

        nlist_tensor = self.cast(nlist_tensor, mstype.int32)
        nlist_tensor = self.reshape(nlist_tensor, (1, 192, 138))

        b_nlist = nlist_tensor > -1
        b_nlist = self.cast(b_nlist, mstype.int32)
        nlist_tensor_r = b_nlist * nlist_tensor
        nlist_tensor_r = self.reshape(nlist_tensor_r, (-1,))

        frames = self.cast(frames, mstype.int32)
        frames = self.expdims(frames, 1)
        frames = self.broadcastto(frames)
        frames = self.reshape(frames, (-1,))

        nlist_tensor_r = nlist_tensor_r + frames
        nlist_tensor_r = self.reshape(nlist_tensor_r, (-1,))

        d_coord_tensor = self.reshape(d_coord_tensor, (-1, 3))
        selected_coord = self.gather(d_coord_tensor, nlist_tensor_r, 0)
        selected_coord = self.reshape(selected_coord, (1, 192, 138, 3))

        coord_tensor_expanded = self.expdims(coord_tensor, 2)
        coord_tensor_expanded = self.broadcastto1(coord_tensor_expanded)

        result_rij_m = selected_coord - coord_tensor_expanded

        b_nlist_expanded = self.expdims(b_nlist, 3)
        b_nlist_expanded = self.broadcastto1(b_nlist_expanded)

        result_rij = result_rij_m * b_nlist_expanded

        return result_rij


class ComputeDescriptor(nn.Cell):
    """compute descriptor."""
    def __init__(self):
        super(ComputeDescriptor, self).__init__()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.rsum = P.ReduceSum()
        self.broadcastto = P.BroadcastTo((1, 192 * 138))
        self.broadcastto1 = P.BroadcastTo((1, 192, 138, 3))
        self.broadcastto2 = P.BroadcastTo((1, 192, 138, 3, 3))
        self.broadcastto3 = P.BroadcastTo((1, 192, 138, 4))
        self.broadcastto4 = P.BroadcastTo((1, 192, 138, 4, 3))

        self.expdims = P.ExpandDims()
        self.concat = P.Concat(axis=3)
        self.gather = P.Gather()
        self.mul = P.Mul()
        self.slice = P.Slice()
        self.square = P.Square()
        self.inv = P.Inv()
        self.sqrt = P.Sqrt()
        self.ones = P.OnesLike()
        self.eye = P.Eye()

    def construct(self, rij_tensor, avg_tensor, std_tensor, nlist_tensor, atype_tensor, r_min=5.8, r_max=6.0):
        """construct function."""
        nlist_tensor = self.reshape(nlist_tensor, (1, 192, 138))
        b_nlist = nlist_tensor > -1
        b_nlist = self.cast(b_nlist, mstype.int32)
        b_nlist_expanded = self.expdims(b_nlist, 3)
        b_nlist_4 = self.broadcastto3(b_nlist_expanded)
        b_nlist_3 = self.broadcastto1(b_nlist_expanded)
        b_nlist_expanded = self.expdims(b_nlist_expanded, 4)
        b_nlist_33 = self.broadcastto2(b_nlist_expanded)

        rij_tensor = rij_tensor + self.cast(1 - b_nlist_3, mstype.float32)

        r_2 = self.square(rij_tensor)
        d_2 = self.rsum(r_2, 3)
        invd_2 = self.inv(d_2)
        invd = self.sqrt(invd_2)
        invd_4 = self.square(invd_2)
        d = invd * d_2
        invd_3 = invd_4 * d

        b_d_1 = self.cast(d < r_max, mstype.int32)
        b_d_2 = self.cast(d < r_min, mstype.int32)
        b_d_3 = self.cast(d >= r_min, mstype.int32)

        du = 1.0 / (r_max - r_min)
        uu = (d - r_min) * du
        vv = uu * uu * uu * (-6 * uu * uu + 15 * uu - 10) + 1
        dd = (3 * uu * uu * (-6 * uu * uu + 15 * uu - 10) + uu * uu * uu * (-12 * uu + 15)) * du

        sw = vv * b_d_3 * b_d_1 + b_d_2
        dsw = dd * b_d_3 * b_d_1

        invd_2_e = self.expdims(invd_2, 3)
        invd_2_e = self.broadcastto1(invd_2_e)
        descrpt_1 = rij_tensor * invd_2_e

        factor0 = invd_3 * sw - invd_2 * dsw
        factor0 = self.expdims(factor0, 3)
        factor0 = self.broadcastto1(factor0)
        descrpt_deriv_0 = rij_tensor * factor0
        descrpt_deriv_0 = descrpt_deriv_0 * b_nlist_3
        descrpt_deriv_0 = self.expdims(descrpt_deriv_0, 3)

        factor1_0 = self.eye(3, 3, mstype.float32)
        factor1_0 = self.expdims(factor1_0, 0)
        factor1_0 = self.expdims(factor1_0, 0)
        factor1_0 = self.expdims(factor1_0, 0)
        factor1_1 = self.expdims(invd_2 * sw, 3)
        factor1_1 = self.expdims(factor1_1, 4)
        descrpt_deriv_1_0 = factor1_0 * factor1_1

        rij_tensor_e1 = self.expdims(rij_tensor, 4)
        rij_tensor_e2 = self.expdims(rij_tensor, 3)
        rij_tensor_e1 = self.broadcastto2(rij_tensor_e1)
        rij_tensor_e2 = self.broadcastto2(rij_tensor_e2)

        factor1_3 = self.expdims(2.0 * invd_4 * sw, 3)
        factor1_3 = self.expdims(factor1_3, 4)
        factor1_3 = self.broadcastto2(factor1_3)
        descrpt_deriv_1_1 = factor1_3 * rij_tensor_e1 * rij_tensor_e2

        factor1_4 = self.expdims(invd * dsw, 3)
        factor1_4 = self.expdims(factor1_4, 3)
        factor1_4 = self.broadcastto2(factor1_4)
        descrpt_1_e = self.expdims(descrpt_1, 4)
        descrpt_1_e = self.broadcastto2(descrpt_1_e)
        descrpt_deriv_1_2 = descrpt_1_e * rij_tensor_e2 * factor1_4

        descrpt_deriv_1 = (descrpt_deriv_1_1 - descrpt_deriv_1_0 - descrpt_deriv_1_2) * b_nlist_33

        descrpt_deriv = self.concat((descrpt_deriv_0, descrpt_deriv_1))

        invd_e = self.expdims(invd, 3)
        descrpt = self.concat((invd_e, descrpt_1))
        sw = self.broadcastto3(self.expdims(sw, 3))
        descrpt = descrpt * sw * b_nlist_4

        avg_tensor = self.cast(avg_tensor, mstype.float32)
        std_tensor = self.cast(std_tensor, mstype.float32)

        atype_tensor = self.reshape(atype_tensor, (-1,))
        atype_tensor = self.cast(atype_tensor, mstype.int32)
        avg_tensor = self.gather(avg_tensor, atype_tensor, 0)
        std_tensor = self.gather(std_tensor, atype_tensor, 0)
        avg_tensor = self.reshape(avg_tensor, (1, 192, 138, 4))
        std_tensor = self.reshape(std_tensor, (1, 192, 138, 4))

        std_tensor_2 = self.expdims(std_tensor, 4)
        std_tensor_2 = self.broadcastto4(std_tensor_2)

        descrpt = (descrpt - avg_tensor) / std_tensor
        descrpt_deriv = descrpt_deriv / std_tensor_2

        return descrpt, descrpt_deriv


class DescriptorSeA(nn.Cell):
    def __init__(self):
        super(DescriptorSeA, self).__init__()
        self.compute_rij = ComputeRij()
        self.compute_descriptor = ComputeDescriptor()

    def construct(self, coord, nlist, frames, avg, std, atype):
        rij = self.compute_rij(coord, nlist, frames)
        descrpt, descrpt_deriv = self.compute_descriptor(rij, avg, std, nlist, atype)
        return rij, descrpt, descrpt_deriv

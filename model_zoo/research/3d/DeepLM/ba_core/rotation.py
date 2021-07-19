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
# ===========================================================================
"""rotation func."""
from .utils import ComputeBase


class AngleAxisRotatePoint(ComputeBase):
    """rotate by a specified angle"""
    def construct(self, angle_axis, points):
        """construct"""
        theta2 = self.sum(self.mul(angle_axis, angle_axis), axis=1)
        mask = (theta2 > 0)

        theta = self.sqrt(theta2 + (1 - mask))
        mask = self.reshape(mask, (mask.shape[0], 1))
        mask = self.concat((mask, mask, mask))
        costheta = self.cos(theta)
        sintheta = self.sin(theta)

        theta_inverse = 1.0 / theta

        w_0 = self.mul(angle_axis[:, 0], theta_inverse)
        w_1 = self.mul(angle_axis[:, 1], theta_inverse)
        w_2 = self.mul(angle_axis[:, 2], theta_inverse)

        w_cross_point_0 = self.mul(w_1, points[:, 2]) - self.mul(w_2, points[:, 1])
        w_cross_point_1 = self.mul(w_2, points[:, 0]) - self.mul(w_0, points[:, 2])
        w_cross_point_2 = self.mul(w_0, points[:, 1]) - self.mul(w_1, points[:, 0])

        tmp = self.mul(self.mul(w_0, points[:, 0]) + self.mul(w_1, points[:, 1]) + self.mul(w_2, points[:, 2]),
                       (1.0 - costheta))
        r_0 = self.mul(points[:, 0], costheta) + self.mul(w_cross_point_0, sintheta) + self.mul(w_0, tmp)
        r_1 = self.mul(points[:, 1], costheta) + self.mul(w_cross_point_1, sintheta) + self.mul(w_1, tmp)
        r_2 = self.mul(points[:, 2], costheta) + self.mul(w_cross_point_2, sintheta) + self.mul(w_2, tmp)

        r_0 = self.reshape(r_0, (r_0.shape[0], 1))
        r_1 = self.reshape(r_1, (r_1.shape[0], 1))
        r_2 = self.reshape(r_2, (r_2.shape[0], 1))

        res1 = self.concat((r_0, r_1, r_2))

        w_cross_point_0 = self.mul(angle_axis[:, 1], points[:, 2]) - self.mul(angle_axis[:, 2], points[:, 1])
        w_cross_point_1 = self.mul(angle_axis[:, 2], points[:, 0]) - self.mul(angle_axis[:, 0], points[:, 2])
        w_cross_point_2 = self.mul(angle_axis[:, 0], points[:, 1]) - self.mul(angle_axis[:, 1], points[:, 0])

        r00 = points[:, 0] + w_cross_point_0
        r01 = points[:, 1] + w_cross_point_1
        r02 = points[:, 2] + w_cross_point_2

        r00 = self.reshape(r00, (r00.shape[0], 1))
        r01 = self.reshape(r01, (r01.shape[0], 1))
        r02 = self.reshape(r02, (r02.shape[0], 1))

        res2 = self.concat((r00, r01, r02))
        return self.mul(res1, mask) + self.mul(res2, 1 - mask)

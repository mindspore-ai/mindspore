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
"""DeepLM loss."""
from .rotation import AngleAxisRotatePoint
from .utils import ComputeBase


class Distort(ComputeBase):
    """distort func"""
    def construct(self, xp, yp, cam):
        """construct"""
        l1 = cam[:, 7]
        l2 = cam[:, 8]
        r2 = self.mul(xp, xp) + self.mul(yp, yp)
        distortion = 1.0 + self.mul(r2, (l1 + self.mul(l2, r2)))

        focal = cam[:, 6]
        predicted_x = self.mul(self.mul(-focal, xp), distortion)
        predicted_y = self.mul(self.mul(-focal, yp), distortion)

        return predicted_x, predicted_y


class SnavelyReprojectionError(ComputeBase):
    """projection func"""
    def __init__(self):
        super(SnavelyReprojectionError, self).__init__()
        self.angleAxisRotatePoint = AngleAxisRotatePoint()
        self.distort = Distort()

    def construct(self, points_ob, cameras_ob, features):
        """construct"""
        if len(points_ob.shape) == 3:
            points_ob = points_ob[:, 0, :]
            cameras_ob = cameras_ob[:, 0, :]
        p = self.angleAxisRotatePoint(cameras_ob[:, :3], points_ob)
        p = p + cameras_ob[:, 3:6]

        xp = self.div(p[:, 0], p[:, 2])
        yp = self.div(p[:, 1], p[:, 2])
        predicted_x, predicted_y = self.distort(xp, yp, cameras_ob)

        residual_0 = predicted_x - features[:, 0]
        residual_1 = predicted_y - features[:, 1]
        residual_0 = self.reshape(residual_0, (residual_0.shape[0], 1))
        residual_1 = self.reshape(residual_1, (residual_1.shape[0], 1))

        return self.concat((residual_0, residual_1))

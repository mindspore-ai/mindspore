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
"""DeepLM io."""
import numpy as np


def load_bal_from_file(filename, feature_dim, camera_dim, point_dim, double=True):
    """load ba data"""
    dtype = np.float64 if double else np.float32
    with open(filename, 'r') as f:
        num_cameras, num_points, num_observations = [int(i) for i in f.readline().strip().split()]
        point_indices = []
        cam_indices = []

        t_camera = np.zeros((num_cameras, camera_dim)).astype(dtype)
        t_point = np.zeros((num_points, point_dim)).astype(dtype)
        t_feat = np.zeros((num_observations, feature_dim)).astype(dtype)

        for i in range(num_observations):
            features2d = []
            if i % 1000 == 0:
                print("\r Load observation {} of {}".format(i, num_observations), end="", flush=True)
            cam_idx, point_idx, x, y = f.readline().strip().split()
            point_indices.append(int(point_idx))
            cam_indices.append(int(cam_idx))
            features2d.append(float(x))
            features2d.append(float(y))
            t_feat[i] = (features2d)

        t_point_indices = point_indices
        t_cam_indices = cam_indices

        for i in range(num_cameras):
            camera_paras = []
            for _ in range(camera_dim):
                camera_para = f.readline().strip().split()[0]
                camera_paras.append(float(camera_para))
            t_camera[i] = camera_paras

        for i in range(num_points):
            points3d = []
            for _ in range(point_dim):
                point = f.readline().strip().split()[0]
                points3d.append(float(point))
            t_point[i] = points3d

    return t_point, t_camera, t_feat, t_point_indices, t_cam_indices

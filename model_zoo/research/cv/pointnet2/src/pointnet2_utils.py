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
"""network definition utils"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as P
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import constexpr

from src.layers import Conv2d


@constexpr
def generate_tensor_fps(B, N):
    """generate tensor"""
    farthest = Tensor(np.random.randint(N, size=(B,)), ms.int32)
    return farthest


@constexpr
def generate_tensor_batch_indices(B):
    """generate tensor"""
    return Tensor(np.arange(B), ms.int32)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * P.BatchMatMul()(src, P.Transpose()(dst, (0, 2, 1)))
    dist += P.Reshape()(P.ReduceSum()(src ** 2, -1), (B, N, 1))
    dist += P.Reshape()(P.ReduceSum()(dst ** 2, -1), (B, 1, M))
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, nsample]
    Return:
        new_points:, indexed points data, [B, S, C] or [B, S, nsample, C]
    """
    shape = idx.shape
    batch_indices = generate_tensor_batch_indices(shape[0])
    if len(shape) == 2:
        batch_indices = batch_indices.view(shape[0], 1)
    else:
        batch_indices = batch_indices.view(shape[0], 1, 1)
    batch_indices = batch_indices.expand_as(idx)
    index = P.Concat(-1)((batch_indices.reshape(idx.shape + (1,)), idx.reshape(idx.shape + (1,))))
    new_points = P.GatherNd()(points, index)
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, _ = xyz.shape
    centroids = mnp.zeros((npoint, B), ms.int32)
    distance = mnp.ones((B, N), ms.int32) * 1e9
    farthest = generate_tensor_fps(B, N)
    batch_indices = generate_tensor_batch_indices(B)
    for i in range(npoint):
        centroids = P.Cast()(centroids, ms.float32)
        farthest = P.Cast()(farthest, ms.float32)
        centroids[i] = farthest
        centroids = P.Cast()(centroids, ms.int32)
        farthest = P.Cast()(farthest, ms.int32)
        index = P.Concat(-1)((batch_indices.reshape(batch_indices.shape + (1,)),
                              farthest.reshape(farthest.shape + (1,))))
        centroid = P.GatherNd()(xyz, index).reshape((B, 1, 3))
        dist = P.ReduceSum()((xyz - centroid) ** 2, -1)
        distance = P.Minimum()(distance, dist)
        farthest = P.Argmax()(distance)
    return P.Transpose()(centroids, (1, 0))


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = mnp.arange(0, N, 1, ms.int32).view(1, 1, N)
    group_idx = P.Tile()(group_idx, (B, S, 1))
    sqrdists = square_distance(new_xyz, xyz)

    idx = sqrdists > radius ** 2
    group_idx = P.Select()(idx, -1 * P.OnesLike()(group_idx), group_idx)
    group_idx = P.Cast()(group_idx, ms.float32)
    group_idx, _ = P.TopK()(group_idx, nsample)
    group_idx = P.Cast()(group_idx, ms.int32)

    group_first = group_idx[:, :, 0].view(B, S, 1)
    group_first = P.Tile()(group_first, (1, 1, nsample))  # [B, S, nsample]

    index = group_idx != -1
    group_first = P.Select()(index, -1 * P.OnesLike()(group_first), group_first)
    group_idx = P.Maximum()(group_idx, group_first)

    return group_idx


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    B, N, C = xyz.shape
    grouped_xyz = P.Reshape()(xyz, (B, 1, N, C))
    new_points = P.Concat(-1)((grouped_xyz, P.Reshape()(points, (B, 1, N, -1))))
    return new_points


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, _, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, S)  # [B, S]
    new_xyz = index_points(xyz, fps_idx)  # [B, S, C]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, S, nsample]
    grouped_xyz = index_points(xyz, idx)  # [B, S, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = P.Concat(-1)((grouped_xyz_norm, grouped_points))  # [B, S, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Cell):
    """PointNetSetAbstraction"""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.conv1 = Conv2d(in_channel, mlp[0], 1)
        self.bn1 = nn.BatchNorm2d(mlp[0])
        self.conv2 = Conv2d(mlp[0], mlp[1], 1)
        self.bn2 = nn.BatchNorm2d(mlp[1])
        self.conv3 = Conv2d(mlp[1], mlp[2], 1)
        self.bn3 = nn.BatchNorm2d(mlp[2])

        self.relu = P.ReLU()
        self.transpose = P.Transpose()
        self.reduce_max = P.ReduceMax()

    def construct(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        if points is not None:
            points = self.transpose(points, (0, 2, 1))

        if self.group_all:
            new_points = sample_and_group_all(xyz, points)
            new_xyz = None
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            # new_xyz: sampled points position data, [B, npoint, C]
            # new_points: sampled points data, [B, npoint, nsample, C+D]

        d1, d2, d3, d4 = new_points.shape
        new_points = self.transpose(new_points.reshape((d1, d2 * d3, d4)), (0, 2, 1))
        new_points = self.transpose(new_points.reshape((d1 * d4, d2, d3)), (0, 2, 1)).reshape((d1, d4, d3, d2))

        new_points = self.relu(self.bn1(self.conv1(new_points)))
        new_points = self.relu(self.bn2(self.conv2(new_points)))
        new_points = self.relu(self.bn3(self.conv3(new_points)))

        new_points = self.reduce_max(new_points, 2)

        return new_xyz, new_points

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
"""
Functional Cells to be used.
"""

import mindspore.nn as nn
import mindspore.ops as ops


class GatherFeature(nn.Cell):
    """
    Gather feature at specified position

    Args: None

    Returns:
        Tensor, feature at spectified position
    """

    def __init__(self):
        super(GatherFeature, self).__init__()
        self.tile = ops.Tile()
        self.shape = ops.Shape()
        self.concat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()
        self.gather_nd = ops.GatherNd()

    def construct(self, feat, ind):
        """gather by specified index"""
        # (b, N)->(b*N, 1)
        b, N = self.shape(ind)
        ind = self.reshape(ind, (-1, 1))
        ind_b = nn.Range(0, b, 1)()
        ind_b = self.reshape(ind_b, (-1, 1))
        ind_b = self.tile(ind_b, (1, N))
        ind_b = self.reshape(ind_b, (-1, 1))
        index = self.concat((ind_b, ind))
        # (b, N, 2)
        index = self.reshape(index, (b, N, -1))
        # (b, N, c)
        feat = self.gather_nd(feat, index)
        return feat


class TransposeGatherFeature(nn.Cell):
    """
    Transpose and gather feature at specified position

    Args: None

    Returns:
        Tensor, feature at spectified position
    """

    def __init__(self):
        super(TransposeGatherFeature, self).__init__()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.perm_list = (0, 2, 3, 1)
        self.gather_feat = GatherFeature()

    def construct(self, feat, ind):
        """(b, c, h, w)->(b, h*w, c)"""
        feat = self.transpose(feat, self.perm_list)
        b, _, _, c = self.shape(feat)
        feat = self.reshape(feat, (b, -1, c))
        # (b, N, c)
        feat = self.gather_feat(feat, ind)
        return feat


class Sigmoid(nn.Cell):
    """
    Sigmoid and then Clip by value

    Args: None

    Returns:
        Tensor, feature after sigmoid and clip.
    """

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.sigmoid = nn.Sigmoid()
        self.clip_by_value = ops.clip_by_value

    def construct(self, x, min_value=1e-4, max_value=1 - 1e-4):
        """Sigmoid and then Clip by value"""
        x = self.sigmoid(x)
        dt = self.dtype(x)
        x = self.clip_by_value(x, self.cast(ops.tuple_to_array((min_value,)), dt),
                               self.cast(ops.tuple_to_array((max_value,)), dt))
        return x

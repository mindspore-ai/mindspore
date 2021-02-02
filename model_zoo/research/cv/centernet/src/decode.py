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
Decode from heads for evaluation
"""

import numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from .utils import GatherFeature, TransposeGatherFeature


class NMS(nn.Cell):
    """
    Non-maximum suppression

    Args:
        kernel(int): Maxpooling kernel size. Default: 3.
        enable_nms_fp16(bool): Use float16 data for max_pool, adaption for CPU. Default: True.

    Returns:
        Tensor, heatmap after non-maximum suppression.
    """
    def __init__(self, kernel=3, enable_nms_fp16=True):
        super(NMS, self).__init__()
        self.pad = (kernel - 1) // 2
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.equal = ops.Equal()
        self.max_pool = nn.MaxPool2d(kernel, stride=1, pad_mode="same")
        self.enable_fp16 = enable_nms_fp16

    def construct(self, heat):
        """Non-maximum suppression"""
        dtype = self.dtype(heat)
        if self.enable_fp16:
            heat = self.cast(heat, mstype.float16)
            heat_max = self.max_pool(heat)
            keep = self.equal(heat, heat_max)
            keep = self.cast(keep, dtype)
            heat = self.cast(heat, dtype)
        else:
            heat_max = self.max_pool(heat)
            keep = self.equal(heat, heat_max)
        heat = heat * keep
        return heat


class GatherTopK(nn.Cell):
    """
    Gather topk features through all channels

    Args: None

    Returns:
        Tuple of Tensors, top_k scores, indexes, category ids, and the indexes in height and width direcction.
    """
    def __init__(self):
        super(GatherTopK, self).__init__()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.topk = ops.TopK(sorted=True)
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.gather_feat = GatherFeature()
        self.mod = ops.Mod()
        self.div = ops.Div()

    def construct(self, scores, K=40):
        """gather top_k"""
        b, c, _, w = self.shape(scores)
        scores = self.reshape(scores, (b, c, -1))
        # (b, c, K)
        topk_scores, topk_inds = self.topk(scores, K)
        topk_ys = self.div(topk_inds, w)
        topk_xs = self.mod(topk_inds, w)
        # (b, K)
        topk_score, topk_ind = self.topk(self.reshape(topk_scores, (b, -1)), K)
        topk_clses = self.cast(self.div(topk_ind, K), self.dtype(scores))
        topk_inds = self.gather_feat(self.reshape(topk_inds, (b, -1, 1)), topk_ind)
        topk_inds = self.reshape(topk_inds, (b, K))
        topk_ys = self.gather_feat(self.reshape(topk_ys, (b, -1, 1)), topk_ind)
        topk_ys = self.cast(self.reshape(topk_ys, (b, K)), self.dtype(scores))
        topk_xs = self.gather_feat(self.reshape(topk_xs, (b, -1, 1)), topk_ind)
        topk_xs = self.cast(self.reshape(topk_xs, (b, K)), self.dtype(scores))
        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


class GatherTopKChannel(nn.Cell):
    """
    Gather topk features of each channel.

    Args: None

    Returns:
        Tuple of Tensors, top_k scores, indexes, and the indexes in height and width direcction respectively.
    """
    def __init__(self):
        super(GatherTopKChannel, self).__init__()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.topk = ops.TopK(sorted=True)
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.mod = ops.Mod()
        self.div = ops.Div()

    def construct(self, scores, K=40):
        b, c, _, w = self.shape(scores)
        scores = self.reshape(scores, (b, c, -1))
        topk_scores, topk_inds = self.topk(scores, K)
        topk_ys = self.div(topk_inds, w)
        topk_xs = self.mod(topk_inds, w)
        topk_ys = self.cast(topk_ys, self.dtype(scores))
        topk_xs = self.cast(topk_xs, self.dtype(scores))
        return topk_scores, topk_inds, topk_ys, topk_xs


class GatherFeatureByInd(nn.Cell):
    """
    Gather features by index

    Args:
        enable_cpu_gather (bool): Use cpu operator GatherD to gather feature or not, adaption for CPU. Default: True.

    Returns:
        Tensor
    """
    def __init__(self, enable_cpu_gatherd=True):
        super(GatherFeatureByInd, self).__init__()
        self.tile = ops.Tile()
        self.shape = ops.Shape()
        self.concat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()
        self.enable_cpu_gatherd = enable_cpu_gatherd
        if self.enable_cpu_gatherd:
            self.gather_nd = ops.GatherD()
            self.expand_dims = ops.ExpandDims()
        else:
            self.gather_nd = ops.GatherNd()

    def construct(self, feat, ind):
        """gather by index"""
        # feat: b, J, K, N
        # ind:  b, J, K
        b, J, K = self.shape(ind)
        feat = self.reshape(feat, (b, J, K, -1))
        _, _, _, N = self.shape(feat)
        if self.enable_cpu_gatherd:
            # (b, J, K, N)
            index = self.expand_dims(ind, -1)
            index = self.tile(index, (1, 1, 1, N))
            feat = self.gather_nd(feat, 2, index)
        else:
            ind = self.reshape(ind, (-1, 1))
            ind_b = nn.Range(0, b * J, 1)()
            ind_b = self.reshape(ind_b, (-1, 1))
            ind_b = self.tile(ind_b, (1, K))
            ind_b = self.reshape(ind_b, (-1, 1))
            index = self.concat((ind_b, ind))
            # (b*J, K, 2)
            index = self.reshape(index, (-1, K, 2))
            # (b*J, K)
            feat = self.reshape(feat, (-1, K, N))
            feat = self.gather_nd(feat, index)
            feat = self.reshape(feat, (b, J, K, -1))
        return feat


class FlipTensor(nn.Cell):
    """
    Gather flipped tensor.

    Args: None

    Returns:
        Tensor, flipped tensor.
    """
    def __init__(self):
        super(FlipTensor, self).__init__()
        self.half = ops.Split(axis=0, output_num=2)
        self.flip = ops.ReverseV2(axis=[3])
        self.gather_nd = ops.GatherNd()

    def construct(self, feat):
        feat_o, feat_f = self.half(feat)
        output = (feat_o + self.flip(feat_f)) / 2.0
        return output


class GatherFlipFeature(nn.Cell):
    """
    Gather flipped feature by specified index.

    Args: None

    Returns:
        Tensor, flipped feature.
    """
    def __init__(self):
        super(GatherFlipFeature, self).__init__()
        self.gather_nd = ops.GatherNd()
        self.transpose = ops.Transpose()
        self.perm_list = (1, 0, 2, 3)
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()

    def construct(self, feat, index):
        """gather by index"""
        b, J, h, w = self.shape(feat)
        # J, b, h, w
        feat = self.transpose(feat, self.perm_list)
        # J, bhw
        feat = self.reshape(feat, (J, -1))
        index = self.reshape(index, (J, -1))
        # J, bhw
        feat = self.gather_nd(feat, index)
        feat = self.reshape(feat, (J, b, h, w))
        # b, J, h, w
        feat = self.transpose(feat, self.perm_list)
        return feat


class FlipLR(nn.Cell):
    """
    Gather flipped human pose heatmap.

    Args: None

    Returns:
        Tensor, flipped heatmap.
    """
    def __init__(self):
        super(FlipLR, self).__init__()
        self.gather_flip_feat = GatherFlipFeature()
        self.half = ops.Split(axis=0, output_num=2)
        self.flip = ops.ReverseV2(axis=[3])
        self.flip_index = Tensor(np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], np.int32))
        self.gather_nd = ops.GatherNd()

    def construct(self, feat):
        # feat: 2*b, J, h, w
        feat_o, feat_f = self.half(feat)
        # b, J, h, w
        feat_f = self.flip(feat_f)
        feat_f = self.gather_flip_feat(feat_f, self.flip_index)
        output = (feat_o + feat_f) / 2.0
        return output


class FlipLROff(nn.Cell):
    """
    Gather flipped keypoints offset.

    Args: None

    Returns:
        Tensor, flipped keypoints offset.
    """
    def __init__(self):
        super(FlipLROff, self).__init__()
        self.gather_flip_feat = GatherFlipFeature()
        self.flip_index = Tensor(np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], np.int32))
        self.half = ops.Split(axis=0, output_num=2)
        self.split = ops.Split(axis=1, output_num=2)
        self.flip = ops.ReverseV2(axis=[3])
        self.concat = ops.Concat(axis=1)

    def construct(self, kps):
        """flip and gather kps at specified position"""
        # kps: 2b, 2J, h, w
        kps_o, kps_f = self.half(kps)
        # b, 2J, h, w
        kps_f = self.flip(kps_f)
        # b, J, h, w
        kps_ow, kps_oh = self.split(kps_o)
        kps_fw, kps_fh = self.split(kps_f)
        kps_fw = -1.0 * kps_fw
        kps_fw = self.gather_flip_feat(kps_fw, self.flip_index)
        kps_fh = self.gather_flip_feat(kps_fh, self.flip_index)
        kps_w = (kps_ow + kps_fw) / 2.0
        kps_h = (kps_oh + kps_fh) / 2.0
        kps = self.concat((kps_w, kps_h))
        return kps


class MultiPoseDecode(nn.Cell):
    """
    Decode from heads to gather multi-person pose info.

    Args:
        net_config(edict): config info for CenterNet network.
        K(int): maximum objects number. Default: 100.
        enable_nms_fp16(bool): Use float16 data for max_pool, adaption for CPU. Default: True.

    Returns:
        Tensor, multi-objects detections.
    """
    def __init__(self, net_config, K=100, enable_nms_fp16=True):
        super(MultiPoseDecode, self).__init__()
        self.K = K
        self.nms = NMS(enable_nms_fp16=enable_nms_fp16)
        self.shape = ops.Shape()
        self.gather_topk = GatherTopK()
        self.gather_topk_channel = GatherTopKChannel()
        self.gather_by_ind = GatherFeatureByInd()
        self.half = ops.Split(axis=-1, output_num=2)
        self.half_first = ops.Split(axis=0, output_num=2)
        self.split = ops.Split(axis=-1, output_num=4)
        self.flip_lr = FlipLR()
        self.flip_lr_off = FlipLROff()
        self.flip_tensor = FlipTensor()
        self.concat = ops.Concat(axis=1)
        self.concat_a2 = ops.Concat(axis=2)
        self.concat_a3 = ops.Concat(axis=3)
        self.trans_gather_feature = TransposeGatherFeature()
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.add = ops.Add()
        self.dtype = ops.DType()
        self.cast = ops.Cast()
        self.thresh = 0.1
        self.transpose = ops.Transpose()
        self.perm_list = (0, 2, 1, 3)
        self.tile = ops.Tile()
        self.greater = ops.Greater()
        self.square = ops.Square()
        self.sqrt = ops.Sqrt()
        self.reduce_sum = ops.ReduceSum()
        self.min = ops.ArgMinWithValue(axis=3)
        self.max = ops.Maximum()
        self.hm_hp = net_config.hm_hp
        self.dense_hp = net_config.dense_hp
        self.reg_offset = net_config.reg_offset
        self.reg_hp_offset = net_config.reg_hp_offset
        self.hm_hp_ind = 3 if self.hm_hp else 2
        self.reg_ind = self.hm_hp_ind + 1 if self.reg_offset else self.hm_hp_ind
        self.reg_hp_ind = self.reg_ind + 1 if self.reg_hp_offset else self.reg_ind

    def construct(self, feature):
        """gather detections"""
        heat = feature[0]
        K = self.K
        b, _, _, _ = self.shape(heat)
        heat = self.nms(heat)
        scores, inds, clses, ys, xs = self.gather_topk(heat, K=K)
        ys = self.reshape(ys, (b, K, 1))
        xs = self.reshape(xs, (b, K, 1))

        kps = feature[1]
        num_joints = self.shape(kps)[1] / 2
        # (b, K, num_joints*2)
        kps = self.trans_gather_feature(kps, inds)
        kps = self.reshape(kps, (b, K, num_joints, 2))
        kps_w, kps_h = self.half(kps)
        # (b, K, num_joints)
        kps_w = self.reshape(kps_w, (b, K, num_joints))
        kps_h = self.reshape(kps_h, (b, K, num_joints))
        kps_h = self.add(kps_h, ys)
        kps_w = self.add(kps_w, xs)
        kps_w = self.reshape(kps_w, (b, K, num_joints, 1))
        kps_h = self.reshape(kps_h, (b, K, num_joints, 1))
        # (b, K, 2*num_joints)
        kps = self.concat_a3((kps_w, kps_h))
        kps = self.reshape(kps, (b, K, num_joints * 2))

        wh = feature[2]
        wh = self.trans_gather_feature(wh, inds)
        ws, hs = self.half(wh)

        if self.reg_offset:
            reg = feature[self.reg_ind]
            reg = self.trans_gather_feature(reg, inds)
            reg = self.reshape(reg, (b, K, 2))
            reg_w, reg_h = self.half(reg)
            ys = self.add(ys, reg_h)
            xs = self.add(xs, reg_w)
        else:
            ys = ys + 0.5
            xs = xs + 0.5

        bboxes = self.concat_a2((xs - ws / 2, ys - hs / 2, xs + ws / 2, ys + hs / 2))

        if self.hm_hp:
            hm_hp = feature[self.hm_hp_ind]
            hm_hp = self.nms(hm_hp)
            # (b, num_joints, K)
            hm_score, hm_inds, hm_ys, hm_xs = self.gather_topk_channel(hm_hp, K=K)

            if self.reg_hp_offset:
                hp_offset = feature[self.reg_hp_ind]
                hp_offset = self.trans_gather_feature(hp_offset, self.reshape(hm_inds, (b, -1)))
                hp_offset = self.reshape(hp_offset, (b, num_joints, K, 2))
                hp_ws, hp_hs = self.half(hp_offset)
                hp_ws = self.reshape(hp_ws, (b, num_joints, K))
                hp_hs = self.reshape(hp_hs, (b, num_joints, K))
                hm_xs = hm_xs + hp_ws
                hm_ys = hm_ys + hp_hs
            else:
                hm_xs = hm_xs + 0.5
                hm_ys = hm_ys + 0.5

            mask = self.greater(hm_score, self.thresh)
            mask = self.cast(mask, self.dtype(hm_score))
            hm_score = mask * hm_score - (1.0 - mask)
            hm_ys = (1 - mask) * (-10000) + mask * hm_ys
            hm_xs = (1 - mask) * (-10000) + mask * hm_xs

            hm_xs = self.reshape(hm_xs, (b, num_joints, K, 1))
            hm_ys = self.reshape(hm_ys, (b, num_joints, K, 1))
            hm_kps = self.concat_a3((hm_xs, hm_ys)) # (b, J, K, 2)
            reg_hm_kps = self.expand_dims(hm_kps, 2) # (b, J, 1, K, 2)
            reg_hm_kps = self.tile(reg_hm_kps, (1, 1, K, 1, 1)) # (b, J, K, K, 2)

            kps = self.reshape(kps, (b, K, num_joints, 2))
            kps = self.transpose(kps, self.perm_list) # (b, J, K, 2)
            reg_kps = self.expand_dims(kps, 3) # (b, J, K, 1, 2)
            reg_kps = self.tile(reg_kps, (1, 1, 1, K, 1)) # (b, J, K, K, 2)

            dist = self.sqrt(self.reduce_sum(self.square(reg_kps - reg_hm_kps), 4)) # (b, J, K, K)
            min_ind, min_dist = self.min(dist) # (b, J, K)

            hm_score = self.gather_by_ind(hm_score, min_ind) # (b, J, K, 1)
            min_dist = self.expand_dims(min_dist, -1) # (b, J, K, 1)
            hm_kps = self.gather_by_ind(hm_kps, min_ind) # (b, J, K, 2)
            hm_kps_xs, hm_kps_ys = self.half(hm_kps)

            l, t, r, d = self.split(bboxes)
            l = self.tile(self.reshape(l, (b, 1, K, 1)), (1, num_joints, 1, 1))
            t = self.tile(self.reshape(t, (b, 1, K, 1)), (1, num_joints, 1, 1))
            r = self.tile(self.reshape(r, (b, 1, K, 1)), (1, num_joints, 1, 1))
            d = self.tile(self.reshape(d, (b, 1, K, 1)), (1, num_joints, 1, 1))

            mask = (self.cast(self.greater(l, hm_kps_xs), self.dtype(hm_score)) +
                    self.cast(self.greater(hm_kps_xs, r), self.dtype(hm_score)) +
                    self.cast(self.greater(t, hm_kps_ys), self.dtype(hm_score)) +
                    self.cast(self.greater(hm_kps_ys, d), self.dtype(hm_score)) +
                    self.cast(self.greater(self.thresh, hm_score), self.dtype(hm_score)) +
                    self.cast(self.greater(min_dist, self.max(d - t, r - l) * 0.3), self.dtype(hm_score)))

            mask = self.cast(self.greater(mask, 0.0), self.dtype(hm_score))

            kps = (1.0 - mask) * hm_kps + mask * kps
            kps = self.reshape(self.transpose(kps, self.perm_list), (b, K, num_joints * 2))

        # scores: (b, K); bboxes: (b, K, 4); kps: (b, K, J * 2); clses: (b, K)
        scores = self.expand_dims(scores, 2)
        clses = self.expand_dims(clses, 2)
        detection = self.concat_a2((bboxes, scores, kps, clses))
        return detection

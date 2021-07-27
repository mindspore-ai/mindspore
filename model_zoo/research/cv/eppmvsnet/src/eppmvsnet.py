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
"""main architecture of EPP-MVSNet"""

import mindspore.nn as nn
import mindspore.common as mstype
from mindspore import Tensor
from mindspore.ops import operations as P
from src.modules import get_depth_values, determine_center_pixel_interval, groupwise_correlation, entropy_num_based, \
    HomoWarp
from src.networks import UNet2D, CostCompression, CoarseStageRegPair, CoarseStageRegFuse, StageRegFuse


class SingleStage(nn.Cell):
    """single stage"""

    def __init__(self, fuse_reg, height, width, entropy_range=False):
        super(SingleStage, self).__init__()
        self.fuse_reg = fuse_reg
        self.entropy_range = entropy_range

        self.transpose = P.Transpose()
        self.zeros = P.Zeros()
        self.linspace = P.LinSpace()
        self.exp = P.Exp()
        self.expand_dims = P.ExpandDims()
        self.squeeze_1 = P.Squeeze(1)
        self.pow = P.Pow()
        self.tile = P.Tile()
        self.homo_warp = HomoWarp(height, width)

    def construct(self, sample, depth_num, depth_start_override=None, depth_interval_override=None,
                  uncertainty_maps=None):
        """construct function of single stage"""
        ref_feat, src_feats, proj_mats = sample
        depth_start = depth_start_override  # n111 or n1hw
        depth_interval = depth_interval_override  # n111

        D = depth_num
        B, C, H, W = ref_feat.shape

        depth_interval = depth_interval.view(B, 1, 1, 1)
        interim_scale = 1

        ref_ncdhw = self.transpose(ref_feat, (0, 2, 3, 1)).view(B, 1, -1, C)
        ref_ncdhw = self.tile(ref_ncdhw, (1, D, 1, 1)).view(B, D, H, W, C)

        pair_results = []  # MVS

        weight_sum = self.zeros((ref_ncdhw.shape[0], 1, 1, ref_ncdhw.shape[2] // interim_scale,
                                 ref_ncdhw.shape[3] // interim_scale), mstype.float32)
        fused_interim = self.zeros((ref_ncdhw.shape[0], 8, ref_ncdhw.shape[1] // interim_scale, ref_ncdhw.shape[2] //
                                    interim_scale, ref_ncdhw.shape[3] // interim_scale), mstype.float32)

        depth_values = get_depth_values(depth_start, D, depth_interval, False)

        for i in range(src_feats.shape[1]):
            src_feat = src_feats[:, i]
            proj_mat = proj_mats[:, i]
            uncertainty_map = uncertainty_maps[i]
            warped_src = self.homo_warp(src_feat, proj_mat, depth_values)
            cost_volume = groupwise_correlation(ref_ncdhw, warped_src, 8, 1)

            interim = cost_volume
            heads = [uncertainty_map]

            weight = self.expand_dims(self.exp(-heads[0]), 2)
            weight_sum = weight_sum + weight
            fused_interim = fused_interim + interim * weight

        fused_interim /= weight_sum
        est_depth, prob_map, prob_volume = self.fuse_reg(fused_interim, depth_values)

        if self.entropy_range:
            # mean entropy
            entropy_num = entropy_num_based(prob_volume, dim=1, depth_num=D, keepdim=True)
            conf_range = self.pow(D, entropy_num).view(B, -1).mean()
            return est_depth, prob_map, pair_results, conf_range  # MVS
        return est_depth, prob_map, pair_results  # MVS


class SingleStageP1(nn.Cell):
    """part1 of single stage 1"""

    def __init__(self, depth_number=32):
        super(SingleStageP1, self).__init__()
        self.transpose = P.Transpose()
        self.zeros = P.Zeros()
        self.linspace = P.LinSpace()
        self.expand_dims = P.ExpandDims()
        self.squeeze_1 = P.Squeeze(1)
        self.squeeze_last = P.Squeeze(-1)
        self.tile = P.Tile()

        self.zero = Tensor(0, mstype.float32)
        self.compression_ratio = 1
        self.depth_number = depth_number - self.compression_ratio + 1
        self.D = Tensor(self.depth_number - 1, mstype.float32)

    def construct(self, sample, depth_num, depth_start_override=None, depth_interval_override=None, timing=True):
        """construct function of part1 of single stage 1"""
        ref_feat, _, _ = sample
        depth_start = depth_start_override  # n111 or n1hw
        depth_interval = depth_interval_override  # n111

        depth_num *= self.compression_ratio
        depth_interval /= self.compression_ratio

        D = depth_num
        B = ref_feat.shape[0]

        depth_interval = depth_interval.view(B, 1, 1, 1)
        depth_start = depth_start.reshape(B, 1, 1, 1)

        h, w = ref_feat.shape[-2:]
        depth_end = depth_start + (D - 1) * depth_interval
        inverse_depth_interval = (1 / depth_start - 1 / depth_end) / (D - self.compression_ratio)
        depth_values = 1 / depth_end + inverse_depth_interval * \
                       self.linspace(self.zero, self.D, self.depth_number)  # (D)
        depth_values = 1.0 / depth_values
        single_depth_value = depth_values.reshape(B, self.depth_number, 1, 1)
        depth_values = self.tile(single_depth_value, (1, 1, h, w))
        single_depth_value = single_depth_value[:, ::self.compression_ratio, :, :]

        return depth_values, single_depth_value


class SingleStageP3(nn.Cell):
    """part3 of single stage 1"""

    def __init__(self, entropy_range=False, compression_ratio=5):
        super(SingleStageP3, self).__init__()
        self.pair_reg = CoarseStageRegPair()
        self.fuse_reg = CoarseStageRegFuse()
        self.entropy_range = entropy_range
        self.compression_ratio = compression_ratio

        self.transpose = P.Transpose()
        self.zeros = P.Zeros()
        self.linspace = P.LinSpace()
        self.expand_dims = P.ExpandDims()
        self.squeeze_1 = P.Squeeze(1)
        self.squeeze_last = P.Squeeze(-1)
        self.tile = P.Tile()
        self.exp = P.Exp()
        self.pow = P.Pow()

    def construct(self, cost_volume_list, depth_values, sample, depth_num):
        """construct function"""
        ref_feat, _, _ = sample

        B, _, _, _ = ref_feat.shape

        d_scale = 1
        interim_scale = 1

        ref_ncdhw = self.expand_dims(ref_feat, 2)
        pair_results = []  # MVS

        weight_sum = self.zeros((ref_ncdhw.shape[0], 1, 1, ref_ncdhw.shape[3] // interim_scale,
                                 ref_ncdhw.shape[4] // interim_scale), mstype.float32)
        fused_interim = self.zeros((ref_ncdhw.shape[0], 8, depth_num // d_scale,
                                    ref_ncdhw.shape[3] // interim_scale, ref_ncdhw.shape[4] // interim_scale),
                                   mstype.float32)

        for i in range(cost_volume_list.shape[1]):
            cost_volume = cost_volume_list[:, i]

            interim, est_depth, uncertainty_map, occ = self.pair_reg(cost_volume, depth_values)
            pair_results.append([est_depth, [uncertainty_map, occ]])

            weight = self.expand_dims(self.exp(-uncertainty_map), 2)
            weight_sum = weight_sum + weight
            fused_interim = fused_interim + interim * weight
        fused_interim /= weight_sum
        est_depth, prob_map, prob_volume = self.fuse_reg(fused_interim, depth_values)
        if self.entropy_range:
            # mean entropy
            entropy_num = entropy_num_based(prob_volume, dim=1, depth_num=depth_num, keepdim=True)
            conf_range = self.pow(depth_num, entropy_num).view(B, -1).mean()
            return est_depth, prob_map, pair_results, conf_range  # MVS
        return est_depth, prob_map, pair_results  # MVS


class SingleStageP2_S1(nn.Cell):
    """0 interpolation, part2 of single stage 1"""

    def __init__(self, cost_compression, height=32, width=40):
        super(SingleStageP2_S1, self).__init__()
        self.cost_compression = cost_compression

        self.transpose = P.Transpose()
        self.zeros = P.Zeros()
        self.linspace = P.LinSpace()
        self.expand_dims = P.ExpandDims()
        self.squeeze_1 = P.Squeeze(1)
        self.squeeze_last = P.Squeeze(-1)
        self.tile = P.Tile()
        self.homo_warp = HomoWarp(height, width)
        self.stack = P.Stack(1)

    def construct(self, sample, depth_num, depth_start_override=None, depth_interval_override=None, depth_values=None,
                  idx=None):
        """construct function"""
        ref_feat, src_feats, proj_mats = sample

        compression_ratio = 1
        depth_num *= compression_ratio

        B, C, H, W = ref_feat.shape

        ref_ncdhw = self.transpose(ref_feat, (0, 2, 3, 1)).view(B, 1, -1, C)
        ref_ncdhw = self.tile(ref_ncdhw, (1, 32, 1, 1)).view(B, 32, H, W, C)

        src_feat = src_feats[:, idx]
        proj_mat = proj_mats[:, idx]
        src_depth_values = depth_values

        warped_src = self.homo_warp(src_feat, proj_mat, src_depth_values)
        cost_volume = groupwise_correlation(ref_ncdhw, warped_src, 8, 1)
        # dynamic max pool
        cost_volume = self.cost_compression(cost_volume)
        return cost_volume


class SingleStageP2_S3(nn.Cell):
    """2 interpolation, part2 of single stage 1"""

    def __init__(self, cost_compression, height=64, width=80, depth_number=96, depth_ratio=3, compression_ratio=5):
        super(SingleStageP2_S3, self).__init__()
        self.cost_compression = cost_compression

        self.transpose = P.Transpose()
        self.zeros = P.Zeros()
        self.linspace = P.LinSpace()
        self.expand_dims = P.ExpandDims()
        self.squeeze_1 = P.Squeeze(1)
        self.squeeze_last = P.Squeeze(-1)
        self.tile = P.Tile()
        self.linspace = P.LinSpace()
        self.cat1 = P.Concat(axis=1)
        self.stack = P.Stack(1)

        self.homo_warp = HomoWarp(height, width)
        self.max_pool = P.MaxPool3D(kernel_size=(3, 1, 1), strides=(3, 1, 1))

        self.zero = Tensor(0, mstype.float32)
        self.src_compression_ratio = depth_ratio
        self.depth_number = depth_number - depth_ratio + 1
        self.src_D = Tensor(self.depth_number, mstype.float32)
        self.compression_ratio = compression_ratio

    def construct(self, sample, depth_num, depth_start_override=None, depth_interval_override=None, depth_values=None,
                  idx=None):
        """construct function"""
        ref_feat, src_feats, proj_mats = sample
        depth_start = depth_start_override  # n111 or n1hw
        depth_interval = depth_interval_override  # n111

        depth_num *= self.compression_ratio
        depth_interval /= self.compression_ratio

        D = depth_num
        B, C, H, W = ref_feat.shape

        depth_interval = depth_interval.view(B, 1, 1, 1)
        depth_start = depth_start.reshape(B, 1, 1, 1)
        depth_end = depth_start + (D - 1) * depth_interval

        ref_ncdhw = self.transpose(ref_feat, (0, 2, 3, 1)).view(B, 1, -1, C)
        ref_ncdhw = self.tile(ref_ncdhw, (1, 96, 1, 1)).view(B, 96, H, W, C)

        src_feat = src_feats[:, idx]
        proj_mat = proj_mats[:, idx]
        src_D = D // self.compression_ratio * self.src_compression_ratio
        src_inverse_depth_interval = (1 / depth_start - 1 / depth_end) / (src_D - self.src_compression_ratio)
        src_depth_values = 1 / depth_end + src_inverse_depth_interval * \
                           self.linspace(self.zero, self.src_D, self.depth_number)  # (D)
        src_depth_values = 1.0 / src_depth_values

        src_depth_values = src_depth_values.view(B, -1, 1, 1)

        end_interval = self.expand_dims(src_depth_values[:, 1, :, :], 1) - self.expand_dims(
            src_depth_values[:, 0, :, :], 1)
        end_interpolation = self.expand_dims(src_depth_values[:, 0, :, :], 1) - end_interval
        start_interval = self.expand_dims(src_depth_values[:, -2, :, :], 1) - self.expand_dims(
            src_depth_values[:, -1, :, :], 1)
        start_interpolation = self.expand_dims(src_depth_values[:, -1, :, :], 1) - start_interval

        src_depth_values = self.cat1((end_interpolation, src_depth_values, start_interpolation))
        src_depth_values = self.tile(src_depth_values, (1, 1, H, W))
        warped_src = self.homo_warp(src_feat, proj_mat, src_depth_values)
        cost_volume = groupwise_correlation(ref_ncdhw, warped_src, 8, 1)

        # dynamic max pool
        cost_volume = self.cost_compression(cost_volume)
        cost_volume = self.max_pool(cost_volume)
        return cost_volume


class EPPMVSNet(nn.Cell):
    """EPP-MVSNet"""

    def __init__(self, n_depths, interval_ratios, entropy_range=False, shrink_ratio=1,
                 height=None, width=None, distance=0.5):
        super(EPPMVSNet, self).__init__()

        self.feat_ext = UNet2D()
        self.n_depths = n_depths
        self.interval_ratios = interval_ratios
        self.entropy_range = entropy_range
        # hyper parameter used in entropy-based adjustment
        self.shrink_ratio = shrink_ratio
        self.distance = distance

        self.stage1_p1 = SingleStageP1()
        self.stage1_p3 = SingleStageP3(entropy_range=self.entropy_range)

        self.cost_compression = CostCompression()

        self.stage1_p2_s1 = SingleStageP2_S1(self.cost_compression, height=height // 8, width=width // 8)
        self.stage1_p2_s3 = SingleStageP2_S3(self.cost_compression, height=height // 8, width=width // 8)

        self.fuse_reg_2 = StageRegFuse("./ckpts/stage2_reg_fuse.ckpt")
        self.fuse_reg_3 = StageRegFuse("./ckpts/stage3_reg_fuse.ckpt")

        self.stage2 = SingleStage(self.fuse_reg_2, height // 4, width // 4, entropy_range=self.entropy_range)
        self.stage3 = SingleStage(self.fuse_reg_3, height // 2, width // 2, entropy_range=self.entropy_range)

        self.eppmvsnet_p1 = EPPMVSNetP1(self.feat_ext, self.stage1_p1, n_depths=n_depths,
                                        interval_ratios=interval_ratios)
        self.eppmvsnet_p3 = EPPMVSNetP3(self.stage1_p3, self.stage2, self.stage3, n_depths=n_depths,
                                        interval_ratios=interval_ratios, height=height, width=width,
                                        entropy_range=self.entropy_range, shrink_ratio=self.shrink_ratio)

    def construct(self, imgs, proj_mats=None, depth_start=None, depth_interval=None):
        """construct function"""
        feat_pack_1, feat_pack_2, feat_pack_3, depth_values_stage1, pixel_distances \
            = self.eppmvsnet_p1(imgs, proj_mats, depth_start, depth_interval)

        cost_volume_list = []
        ref_feat_1, srcs_feat_1 = feat_pack_1[:, 0], feat_pack_1[:, 1:]

        for i in range(feat_pack_1.shape[1] - 1):
            pixel_distance = P.Squeeze()(pixel_distances[i])

            if pixel_distance < self.distance * 3:
                cost_volume_1 = self.stage1_p2_s1([ref_feat_1, srcs_feat_1, proj_mats[:, :, 2]],
                                                  self.n_depths[0], depth_start,
                                                  depth_interval * self.interval_ratios[0],
                                                  depth_values_stage1, i)

                cost_volume_list.append(cost_volume_1)
            else:
                cost_volume_3 = self.stage1_p2_s3([ref_feat_1, srcs_feat_1, proj_mats[:, :, 2]],
                                                  self.n_depths[0], depth_start,
                                                  depth_interval * self.interval_ratios[0],
                                                  depth_values_stage1, i)
                cost_volume_list.append(cost_volume_3)
        cost_volume_list = P.Stack(1)(cost_volume_list)
        results = self.eppmvsnet_p3(feat_pack_1, feat_pack_2, feat_pack_3, depth_values_stage1, cost_volume_list,
                                    proj_mats, depth_start, depth_interval)
        return results


class EPPMVSNetP1(nn.Cell):
    """EPPMVSNet part1"""

    def __init__(self, feat_ext, stage1_p1, n_depths, interval_ratios, entropy_range=False):
        super(EPPMVSNetP1, self).__init__()
        self.n_depths = n_depths
        self.interval_ratios = interval_ratios
        self.entropy_range = entropy_range
        # hyper parameter used in entropy-based adjustment
        self.shrink_ratio = 1

        self.feat_ext = feat_ext
        self.stage1_p1 = stage1_p1

        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()

    def construct(self, imgs, proj_mats=None, depth_start=None, depth_interval=None):
        """construct function"""
        B, V, _, H, W = imgs.shape
        imgs = imgs.reshape(B * V, 3, H, W)
        feat_pack_1, feat_pack_2, feat_pack_3 = self.feat_ext(imgs)
        feat_pack_1 = feat_pack_1.view(B, V, *feat_pack_1.shape[1:])  # (B, V, C, h, w)
        feat_pack_2 = feat_pack_2.view(B, V, *feat_pack_2.shape[1:])  # (B, V, C, h, w)
        feat_pack_3 = feat_pack_3.view(B, V, *feat_pack_3.shape[1:])  # (B, V, C, h, w)

        ref_feat_1, srcs_feat_1 = feat_pack_1[:, 0], feat_pack_1[:, 1:]

        depth_values_stage1, single_depth_values_stage1 = self.stage1_p1([ref_feat_1, srcs_feat_1, proj_mats[:, :, 2]],
                                                                         depth_num=self.n_depths[0],
                                                                         depth_start_override=depth_start,
                                                                         depth_interval_override=depth_interval *
                                                                         self.interval_ratios[0])

        _, src_feats, proj_mats = [ref_feat_1, srcs_feat_1, proj_mats[:, :, 2]]
        pixel_distances = []
        for i in range(src_feats.shape[1]):
            src_feat = src_feats[:, i]
            proj_mat = proj_mats[:, i]
            pixel_distance = determine_center_pixel_interval(src_feat, proj_mat, single_depth_values_stage1)
            pixel_distances.append(pixel_distance)
        return feat_pack_1, feat_pack_2, feat_pack_3, depth_values_stage1, pixel_distances


class EPPMVSNetP3(nn.Cell):
    """EPPMVSNet part3"""

    def __init__(self, stage1_p3, stage2, stage3, n_depths, interval_ratios, entropy_range=False,
                 shrink_ratio=1, height=None, width=None):
        super(EPPMVSNetP3, self).__init__()
        self.n_depths = n_depths
        self.interval_ratios = interval_ratios
        self.entropy_range = entropy_range
        self.shrink_ratio = shrink_ratio

        self.stage1_p3 = stage1_p3
        self.stage2 = stage2
        self.stage3 = stage3

        self.height = height
        self.width = width

    def construct(self, feat_pack_1, feat_pack_2, feat_pack_3, depth_values_stage1, cost_volume_list_stage1,
                  proj_mats=None, depth_start=None, depth_interval=None):
        """construct function"""
        H = self.height
        W = self.width

        ref_feat_1, srcs_feat_1 = feat_pack_1[:, 0], feat_pack_1[:, 1:]
        if self.entropy_range:
            est_depth_1, _, pair_results_1, conf_range_1 = self.stage1_p3(cost_volume_list_stage1, depth_values_stage1,
                                                                          [ref_feat_1, srcs_feat_1, proj_mats[:, :, 2]],
                                                                          self.n_depths[0])
            stage2_conf_interval = self.shrink_ratio * conf_range_1 / self.n_depths[0] * (
                depth_interval * self.interval_ratios[0] * self.n_depths[0]) / self.n_depths[1]
        else:
            est_depth_1, _, pair_results_1 = self.stage1_p3(cost_volume_list_stage1, depth_values_stage1,
                                                            [ref_feat_1, srcs_feat_1, proj_mats[:, :, 2]],
                                                            self.n_depths[0])
            stage2_conf_interval = None
        uncertainty_maps_1, uncertainty_maps_2 = [], []
        for pair_result in pair_results_1:
            uncertainty_maps_1.append(pair_result[1][0])
        for uncertainty_map in uncertainty_maps_1:
            uncertainty_maps_2.append(P.ResizeBilinear((H // 4, W // 4), False)(uncertainty_map))

        ref_feat_2, srcs_feat_2 = feat_pack_2[:, 0], feat_pack_2[:, 1:]
        depth_start_2 = P.ResizeBilinear((H // 4, W // 4), False)(est_depth_1)

        if self.entropy_range:
            est_depth_2, _, _, conf_range_2 = self.stage2([ref_feat_2, srcs_feat_2, proj_mats[:, :, 1]],
                                                          depth_num=self.n_depths[1],
                                                          depth_start_override=depth_start_2,
                                                          depth_interval_override=stage2_conf_interval,
                                                          uncertainty_maps=uncertainty_maps_2)
            stage3_conf_interval = self.shrink_ratio * conf_range_2 / self.n_depths[1] * (
                stage2_conf_interval * self.n_depths[1]) / self.n_depths[2]
        else:
            est_depth_2, _, _ = self.stage2([ref_feat_2, srcs_feat_2, proj_mats[:, :, 1]],
                                            depth_num=self.n_depths[1], depth_start_override=depth_start_2,
                                            depth_interval_override=depth_interval * self.interval_ratios[1],
                                            uncertainty_maps=uncertainty_maps_2)
            stage3_conf_interval = None
        uncertainty_maps_3 = []
        for uncertainty_map in uncertainty_maps_2:
            uncertainty_maps_3.append(P.ResizeBilinear((H // 2, W // 2), False)(uncertainty_map))

        ref_feat_3, srcs_feat_3 = feat_pack_3[:, 0], feat_pack_3[:, 1:]
        depth_start_3 = P.ResizeBilinear((H // 2, W // 2), False)(est_depth_2)

        if self.entropy_range:
            est_depth_3, prob_map_3, _, _ = self.stage3([ref_feat_3, srcs_feat_3, proj_mats[:, :, 0]],
                                                        depth_num=self.n_depths[2],
                                                        depth_start_override=depth_start_3,
                                                        depth_interval_override=stage3_conf_interval,
                                                        uncertainty_maps=uncertainty_maps_3)
        else:
            est_depth_3, prob_map_3, _ = self.stage3([ref_feat_3, srcs_feat_3, proj_mats[:, :, 0]],
                                                     depth_num=self.n_depths[2],
                                                     depth_start_override=depth_start_3,
                                                     depth_interval_override=depth_interval * self.interval_ratios[2],
                                                     uncertainty_maps=uncertainty_maps_3)
        refined_depth = est_depth_3
        return refined_depth, prob_map_3

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
"""sub-networks of EPP-MVSNet"""

import numpy as np
import mindspore
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter
from src.modules import depth_regression, soft_argmin, entropy


class BasicBlockA(nn.Cell):
    """BasicBlockA"""

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlockA, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, pad_mode="pad")
        self.conv2d_1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, pad_mode="valid")
        self.batchnorm2d_2 = nn.BatchNorm2d(out_channels, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.batchnorm2d_3 = nn.BatchNorm2d(out_channels, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_4 = nn.ReLU()
        self.conv2d_5 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=(1, 1, 1, 1), pad_mode="pad")
        self.batchnorm2d_6 = nn.BatchNorm2d(out_channels, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_8 = nn.ReLU()

    def construct(self, x):
        """construct"""
        x1 = self.conv2d_0(x)
        x1 = self.batchnorm2d_2(x1)
        x1 = self.relu_4(x1)
        x1 = self.conv2d_5(x1)
        x1 = self.batchnorm2d_6(x1)

        res = self.conv2d_1(x)
        res = self.batchnorm2d_3(res)

        out = P.Add()(x1, res)
        out = self.relu_8(out)
        return out


class BasicBlockB(nn.Cell):
    """BasicBlockB"""

    def __init__(self, in_channels, out_channels):
        super(BasicBlockB, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, pad_mode="pad")
        self.batchnorm2d_1 = nn.BatchNorm2d(out_channels, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, pad_mode="pad")
        self.batchnorm2d_4 = nn.BatchNorm2d(out_channels, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_6 = nn.ReLU()

    def construct(self, x):
        """construct"""
        x1 = self.conv2d_0(x)
        x1 = self.batchnorm2d_1(x1)
        x1 = self.relu_2(x1)
        x1 = self.conv2d_3(x1)
        x1 = self.batchnorm2d_4(x1)

        res = x

        out = P.Add()(x1, res)
        out = self.relu_6(out)
        return out


class UNet2D(nn.Cell):
    """UNet2D"""

    def __init__(self):
        super(UNet2D, self).__init__()

        self.conv2d_0 = nn.Conv2d(3, 16, 5, stride=2, padding=2, pad_mode="pad")
        self.batchnorm2d_1 = nn.BatchNorm2d(16, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.leakyrelu_2 = nn.LeakyReLU(alpha=0.009999999776482582)

        self.convblocka_0 = BasicBlockA(16, 32, 1)
        self.convblockb_0 = BasicBlockB(32, 32)

        self.convblocka_1 = BasicBlockA(32, 64, 2)
        self.convblockb_1 = BasicBlockB(64, 64)

        self.convblocka_2 = BasicBlockA(64, 128, 2)
        self.convblockb_2 = BasicBlockB(128, 128)

        self.conv2dbackpropinput_51 = P.Conv2DBackpropInput(64, 3, stride=2, pad=1, pad_mode="pad")
        self.conv2dbackpropinput_51_weight = Parameter(Tensor(
            np.random.uniform(0, 1, (128, 64, 3, 3)).astype(np.float32)))
        self.conv2d_54 = nn.Conv2d(128, 64, 3, stride=1, padding=1, pad_mode="pad")
        self.convblockb_3 = BasicBlockB(64, 64)

        self.conv2dbackpropinput_62 = P.Conv2DBackpropInput(32, 3, stride=2, pad=1, pad_mode="pad")
        self.conv2dbackpropinput_62_weight = Parameter(Tensor(
            np.random.uniform(0, 1, (64, 32, 3, 3)).astype(np.float32)))
        self.conv2d_65 = nn.Conv2d(64, 32, 3, stride=1, padding=1, pad_mode="pad")
        self.convblockb_4 = BasicBlockB(32, 32)

        self.conv2d_52 = nn.Conv2d(128, 32, 3, stride=1, padding=1, pad_mode="pad")
        self.conv2d_63 = nn.Conv2d(64, 32, 3, stride=1, padding=1, pad_mode="pad")
        self.conv2d_73 = nn.Conv2d(32, 32, 3, stride=1, padding=1, pad_mode="pad")

        self.concat = P.Concat(axis=1)

        param_dict = mindspore.load_checkpoint("./ckpts/feat_ext.ckpt")
        params_not_loaded = mindspore.load_param_into_net(self, param_dict, strict_load=True)
        print(params_not_loaded)

    def construct(self, imgs):
        """construct"""
        _, _, h, w = imgs.shape

        x = self.conv2d_0(imgs)
        x = self.batchnorm2d_1(x)
        x = self.leakyrelu_2(x)

        x1 = self.convblocka_0(x)
        x1 = self.convblockb_0(x1)
        x2 = self.convblocka_1(x1)
        x2 = self.convblockb_1(x2)
        x3 = self.convblocka_2(x2)
        x3 = self.convblockb_2(x3)

        x2_upsample = self.conv2dbackpropinput_51(x3, self.conv2dbackpropinput_51_weight,
                                                  (x2.shape[0], x2.shape[1], h // 4, w // 4))
        x2_upsample = self.concat((x2_upsample, x2,))
        x2_upsample = self.conv2d_54(x2_upsample)
        x2_upsample = self.convblockb_3(x2_upsample)

        x1_upsample = self.conv2dbackpropinput_62(x2_upsample, self.conv2dbackpropinput_62_weight,
                                                  (x1.shape[0], x1.shape[1], h // 2, w // 2))
        x1_upsample = self.concat((x1_upsample, x1,))
        x1_upsample = self.conv2d_65(x1_upsample)
        x1_upsample = self.convblockb_4(x1_upsample)

        x3_final = self.conv2d_52(x3)
        x2_final = self.conv2d_63(x2_upsample)
        x1_final = self.conv2d_73(x1_upsample)
        return x3_final, x2_final, x1_final


class ConvBnReLu(nn.Cell):
    """ConvBnReLu"""

    def __init__(self, in_channels, out_channels):
        super(ConvBnReLu, self).__init__()
        self.conv3d_0 = nn.Conv3d(in_channels, out_channels, (3, 1, 1), stride=1, padding=(1, 1, 0, 0, 0, 0),
                                  pad_mode="pad")
        self.batchnorm3d_1 = nn.BatchNorm3d(out_channels, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.leakyrelu_2 = nn.LeakyReLU(alpha=0.009999999776482582)

    def construct(self, x):
        """construct"""
        x = self.conv3d_0(x)
        x = self.batchnorm3d_1(x)
        x = self.leakyrelu_2(x)
        return x


class CostCompression(nn.Cell):
    """CostCompression"""

    def __init__(self):
        super(CostCompression, self).__init__()
        self.basicblock_0 = ConvBnReLu(8, 64)
        self.basicblock_1 = ConvBnReLu(64, 64)
        self.basicblock_2 = ConvBnReLu(64, 8)

        param_dict = mindspore.load_checkpoint("./ckpts/stage1_cost_compression.ckpt")
        params_not_loaded = mindspore.load_param_into_net(self, param_dict, strict_load=True)
        print(params_not_loaded)

    def construct(self, x):
        """construct"""
        x = self.basicblock_0(x)
        x = self.basicblock_1(x)
        x = self.basicblock_2(x)
        return x


class Pseudo3DBlock_A(nn.Cell):
    """Pseudo3DBlock_A"""

    def __init__(self, in_channels, out_channels):
        super(Pseudo3DBlock_A, self).__init__()
        self.conv3d_0 = nn.Conv3d(in_channels, out_channels, (1, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1),
                                  pad_mode="pad")
        self.conv3d_1 = nn.Conv3d(out_channels, out_channels, (3, 1, 1), stride=1, padding=(1, 1, 0, 0, 0, 0),
                                  pad_mode="pad")
        self.batchnorm3d_2 = nn.BatchNorm3d(out_channels, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_3 = nn.ReLU()
        self.conv3d_4 = nn.Conv3d(out_channels, out_channels, (1, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1),
                                  pad_mode="pad")
        self.conv3d_5 = nn.Conv3d(out_channels, out_channels, (3, 1, 1), stride=1, padding=(1, 1, 0, 0, 0, 0),
                                  pad_mode="pad")
        self.batchnorm3d_6 = nn.BatchNorm3d(out_channels, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_8 = nn.ReLU()

    def construct(self, x):
        """construct"""
        x1 = self.conv3d_0(x)
        x1 = self.conv3d_1(x1)
        x1 = self.batchnorm3d_2(x1)
        x1 = self.relu_3(x1)
        x1 = self.conv3d_4(x1)
        x1 = self.conv3d_5(x1)
        x1 = self.batchnorm3d_6(x1)

        res = x

        out = P.Add()(x1, res)
        out = self.relu_8(out)
        return out


class Pseudo3DBlock_B(nn.Cell):
    """Pseudo3DBlock_B"""

    def __init__(self):
        super(Pseudo3DBlock_B, self).__init__()
        self.conv3d_0 = nn.Conv3d(8, 8, (1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 1, 1, 1, 1), pad_mode="pad")
        self.conv3d_1 = nn.Conv3d(8, 16, (1, 1, 1), stride=2, padding=0, pad_mode="valid")
        self.conv3d_2 = nn.Conv3d(8, 16, (3, 1, 1), stride=(2, 1, 1), padding=(1, 1, 0, 0, 0, 0), pad_mode="pad")
        self.batchnorm3d_3 = nn.BatchNorm3d(16, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.batchnorm3d_4 = nn.BatchNorm3d(16, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_5 = nn.ReLU()
        self.conv3d_6 = nn.Conv3d(16, 16, (1, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1), pad_mode="pad")
        self.conv3d_7 = nn.Conv3d(16, 16, (3, 1, 1), stride=1, padding=(1, 1, 0, 0, 0, 0), pad_mode="pad")
        self.batchnorm3d_8 = nn.BatchNorm3d(16, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.relu_10 = nn.ReLU()

    def construct(self, x):
        """construct"""
        x1 = self.conv3d_0(x)
        x1 = self.conv3d_2(x1)
        x1 = self.batchnorm3d_4(x1)
        x1 = self.relu_5(x1)
        x1 = self.conv3d_6(x1)
        x1 = self.conv3d_7(x1)
        x1 = self.batchnorm3d_8(x1)

        res = self.conv3d_1(x)
        res = self.batchnorm3d_3(res)

        out = P.Add()(x1, res)
        out = self.relu_10(out)
        return out


class CoarseStageRegFuse(nn.Cell):
    """CoarseStageRegFuse"""

    def __init__(self):
        super(CoarseStageRegFuse, self).__init__()
        self.basicblocka_0 = Pseudo3DBlock_A(8, 8)
        self.basicblockb_0 = Pseudo3DBlock_B()
        self.conv3dtranspose_21 = nn.Conv3dTranspose(16, 8, 3, stride=2, padding=1, pad_mode="pad", output_padding=1)

        self.conv3d_23 = nn.Conv3d(16, 8, (1, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1), pad_mode="pad")
        self.conv3d_24 = nn.Conv3d(8, 8, (3, 1, 1), stride=1, padding=(1, 1, 0, 0, 0, 0), pad_mode="pad")
        self.conv3d_25 = nn.Conv3d(8, 1, 3, stride=1, padding=1, pad_mode="pad")

        self.concat_1 = P.Concat(axis=1)
        self.squeeze_1 = P.Squeeze(axis=1)

        param_dict = mindspore.load_checkpoint("./ckpts/stage1_reg_fuse.ckpt")
        params_not_loaded = mindspore.load_param_into_net(self, param_dict, strict_load=True)
        print(params_not_loaded)

    def construct(self, fused_interim, depth_values):
        """construct"""
        x1 = self.basicblocka_0(fused_interim)
        x2 = self.basicblockb_0(x1)
        x1_upsample = self.conv3dtranspose_21(x2)

        cost_volume = self.concat_1((x1_upsample, x1))
        cost_volume = self.conv3d_23(cost_volume)
        cost_volume = self.conv3d_24(cost_volume)
        score_volume = self.conv3d_25(cost_volume)

        score_volume = self.squeeze_1(score_volume)

        prob_volume, _, prob_map = soft_argmin(score_volume, dim=1, keepdim=True, window=2)
        est_depth = depth_regression(prob_volume, depth_values, keep_dim=True)
        return est_depth, prob_map, prob_volume


class CoarseStageRegPair(nn.Cell):
    """CoarseStageRegPair"""

    def __init__(self):
        super(CoarseStageRegPair, self).__init__()
        self.basicblocka_0 = Pseudo3DBlock_A(8, 8)
        self.basicblockb_0 = Pseudo3DBlock_B()
        self.conv3dtranspose_21 = nn.Conv3dTranspose(16, 8, 3, stride=2, padding=1, pad_mode="pad", output_padding=1)

        self.concat_22 = P.Concat(axis=1)
        self.conv3d_23 = nn.Conv3d(16, 8, (1, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1), pad_mode="pad")
        self.conv3d_24 = nn.Conv3d(8, 8, (3, 1, 1), stride=1, padding=(1, 1, 0, 0, 0, 0), pad_mode="pad")
        self.conv3d_25 = nn.Conv3d(8, 1, 3, stride=1, padding=1, pad_mode="pad")

        self.conv2d_38 = nn.Conv2d(1, 8, 3, stride=1, padding=1, pad_mode="pad")
        self.batchnorm2d_39 = nn.BatchNorm2d(num_features=8, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.leakyrelu_40 = nn.LeakyReLU(alpha=0.009999999776482582)
        self.conv2d_41 = nn.Conv2d(8, 8, 3, stride=1, padding=1, pad_mode="pad")
        self.batchnorm2d_42 = nn.BatchNorm2d(num_features=8, eps=9.999999747378752e-06, momentum=0.8999999761581421)
        self.leakyrelu_43 = nn.LeakyReLU(alpha=0.009999999776482582)
        self.conv2d_45 = nn.Conv2d(8, 1, 3, stride=1, padding=1, pad_mode="pad")
        self.conv2d_46 = nn.Conv2d(8, 1, 3, stride=1, padding=1, pad_mode="pad")

        self.concat_1 = P.Concat(axis=1)
        self.squeeze_1 = P.Squeeze(axis=1)

        param_dict = mindspore.load_checkpoint("./ckpts/stage1_reg_pair.ckpt")
        params_not_loaded = mindspore.load_param_into_net(self, param_dict, strict_load=True)
        print(params_not_loaded)

    def construct(self, cost_volume, depth_values):
        """construct"""
        x1 = self.basicblocka_0(cost_volume)
        x2 = self.basicblockb_0(x1)
        x1_upsample = self.conv3dtranspose_21(x2)

        interim = self.concat_1((x1_upsample, x1))
        interim = self.conv3d_23(interim)
        interim = self.conv3d_24(interim)
        score_volume = self.conv3d_25(interim)

        score_volume = self.squeeze_1(score_volume)
        prob_volume, _ = soft_argmin(score_volume, dim=1, keepdim=True)
        est_depth = depth_regression(prob_volume, depth_values, keep_dim=True)
        entropy_ = entropy(prob_volume, dim=1, keepdim=True)

        x = self.conv2d_38(entropy_)
        x = self.batchnorm2d_39(x)
        x = self.leakyrelu_40(x)
        x = self.conv2d_41(x)
        x = self.batchnorm2d_42(x)
        x = self.leakyrelu_43(x)

        out = P.Add()(x, entropy_)
        uncertainty_map = self.conv2d_45(out)
        occ = self.conv2d_46(out)

        return interim, est_depth, uncertainty_map, occ


class StageRegFuse(nn.Cell):
    """StageRegFuse"""

    def __init__(self, ckpt_path):
        super(StageRegFuse, self).__init__()
        self.basicblocka_0 = Pseudo3DBlock_A(8, 8)
        self.basicblocka_1 = Pseudo3DBlock_A(8, 8)
        self.basicblockb_0 = Pseudo3DBlock_B()
        self.basicblocka_2 = Pseudo3DBlock_A(16, 16)
        self.conv3dtranspose_38 = nn.Conv3dTranspose(16, 8, 3, stride=2, padding=1, pad_mode="pad", output_padding=1)

        self.concat_39 = P.Concat(axis=1)
        self.conv3d_40 = nn.Conv3d(16, 8, (1, 3, 3), stride=1, padding=(0, 0, 1, 1, 1, 1), pad_mode="pad")
        self.conv3d_41 = nn.Conv3d(8, 8, (3, 1, 1), stride=1, padding=(1, 1, 0, 0, 0, 0), pad_mode="pad")
        self.conv3d_42 = nn.Conv3d(8, 1, 3, stride=1, padding=1, pad_mode="pad")

        self.concat_1 = P.Concat(axis=1)
        self.squeeze_1 = P.Squeeze(axis=1)

        param_dict = mindspore.load_checkpoint(ckpt_path)
        params_not_loaded = mindspore.load_param_into_net(self, param_dict, strict_load=True)
        print(params_not_loaded)

    def construct(self, fused_interim, depth_values):
        """construct"""
        x1 = self.basicblocka_0(fused_interim)
        x1 = self.basicblocka_1(x1)
        x2 = self.basicblockb_0(x1)
        x2 = self.basicblocka_2(x2)
        x1_upsample = self.conv3dtranspose_38(x2)

        cost_volume = self.concat_1((x1_upsample, x1))
        cost_volume = self.conv3d_40(cost_volume)
        cost_volume = self.conv3d_41(cost_volume)
        score_volume = self.conv3d_42(cost_volume)

        score_volume = self.squeeze_1(score_volume)

        prob_volume, _, prob_map = soft_argmin(score_volume, dim=1, keepdim=True, window=2)
        est_depth = depth_regression(prob_volume, depth_values, keep_dim=True)
        return est_depth, prob_map, prob_volume

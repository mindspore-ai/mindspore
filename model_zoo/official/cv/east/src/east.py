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
import mindspore as mstype
import mindspore.nn as nn
import mindspore.ops as P


def _conv(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        pad_mode='pad'):
    """Conv2D wrapper."""
    weights = 'ones'
    layers = []
    layers += [nn.Conv2d(in_channels,
                         out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         pad_mode=pad_mode,
                         weight_init=weights,
                         has_bias=False)]
    layers += [nn.BatchNorm2d(out_channels)]
    return nn.SequentialCell(layers)


class VGG16FeatureExtraction(nn.Cell):
    """VGG16FeatureExtraction for deeptext"""

    def __init__(self):
        super(VGG16FeatureExtraction, self).__init__()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv1_1 = _conv(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1)
        self.conv1_2 = _conv(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1)

        self.conv2_1 = _conv(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1)
        self.conv2_2 = _conv(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1)

        self.conv3_1 = _conv(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1)
        self.conv3_2 = _conv(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1)
        self.conv3_3 = _conv(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1)

        self.conv4_1 = _conv(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            padding=1)
        self.conv4_2 = _conv(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            padding=1)
        self.conv4_3 = _conv(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            padding=1)

        self.conv5_1 = _conv(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            padding=1)
        self.conv5_2 = _conv(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            padding=1)
        self.conv5_3 = _conv(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            padding=1)
        self.cast = P.Cast()

    def construct(self, out):
        """ Construction of VGG """
        f_0 = out
        out = self.cast(out, mstype.float32)
        out = self.conv1_1(out)
        out = self.relu(out)
        out = self.conv1_2(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.conv2_1(out)
        out = self.relu(out)
        out = self.conv2_2(out)
        out = self.relu(out)
        out = self.max_pool(out)
        f_2 = out

        out = self.conv3_1(out)
        out = self.relu(out)
        out = self.conv3_2(out)
        out = self.relu(out)
        out = self.conv3_3(out)
        out = self.relu(out)
        out = self.max_pool(out)
        f_3 = out

        out = self.conv4_1(out)
        out = self.relu(out)
        out = self.conv4_2(out)
        out = self.relu(out)
        out = self.conv4_3(out)
        out = self.relu(out)

        out = self.max_pool(out)
        f_4 = out
        out = self.conv5_1(out)
        out = self.relu(out)
        out = self.conv5_2(out)
        out = self.relu(out)
        out = self.conv5_3(out)
        out = self.relu(out)
        out = self.max_pool(out)
        f_5 = out

        return f_0, f_2, f_3, f_4, f_5


class Merge(nn.Cell):
    def __init__(self):
        super(Merge, self).__init__()

        self.conv1 = nn.Conv2d(1024, 128, 1, has_bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            128,
            128,
            3,
            padding=1,
            pad_mode='pad',
            has_bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(384, 64, 1, has_bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(
            64,
            64,
            3,
            padding=1,
            pad_mode='pad',
            has_bias=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(192, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(
            32,
            32,
            3,
            padding=1,
            pad_mode='pad',
            has_bias=True)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(
            32,
            32,
            3,
            padding=1,
            pad_mode='pad',
            has_bias=True)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()
        self.concat = P.Concat(axis=1)

    def construct(self, x, f1, f2, f3, f4):
        img_hight = P.Shape()(x)[2]
        img_width = P.Shape()(x)[3]

        out = P.ResizeBilinear((img_hight / 16, img_width / 16), True)(f4)
        out = self.concat((out, f3))
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.relu2(self.bn2(self.conv2(out)))

        out = P.ResizeBilinear((img_hight / 8, img_width / 8), True)(out)
        out = self.concat((out, f2))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.relu4(self.bn4(self.conv4(out)))

        out = P.ResizeBilinear((img_hight / 4, img_width / 4), True)(out)
        out = self.concat((out, f1))
        out = self.relu5(self.bn5(self.conv5(out)))
        out = self.relu6(self.bn6(self.conv6(out)))

        out = self.relu7(self.bn7(self.conv7(out)))
        return out


class Output(nn.Cell):
    def __init__(self, scope=512):
        super(Output, self).__init__()
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = scope
        self.concat = P.Concat(axis=1)
        self.PI = 3.1415926535898

    def construct(self, x):
        score = self.sigmoid1(self.conv1(x))
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * self.PI
        geo = self.concat((loc, angle))
        return score, geo


class EAST(nn.Cell):
    def __init__(self):
        super(EAST, self).__init__()
        self.extractor = VGG16FeatureExtraction()
        self.merge = Merge()
        self.output = Output()

    def construct(self, x_1):
        f_0, f_1, f_2, f_3, f_4 = self.extractor(x_1)
        x_1 = self.merge(f_0, f_1, f_2, f_3, f_4)
        score, geo = self.output(x_1)

        return score, geo


class DiceCoefficient(nn.Cell):
    def __init__(self):
        super(DiceCoefficient, self).__init__()
        self.sum = P.ReduceSum()
        self.eps = 1e-5

    def construct(self, true_cls, pred_cls):
        intersection = self.sum(true_cls * pred_cls, ())
        union = self.sum(true_cls, ()) + self.sum(pred_cls, ()) + self.eps
        loss = 1. - (2 * intersection / union)

        return loss


class MyMin(nn.Cell):
    def __init__(self):
        super(MyMin, self).__init__()
        self.abs = P.Abs()

    def construct(self, a, b):
        return (a + b - self.abs(a - b)) / 2


class EastLossBlock(nn.Cell):
    def __init__(self):
        super(EastLossBlock, self).__init__()
        self.split = P.Split(1, 5)
        self.min = MyMin()
        self.log = P.Log()
        self.cos = P.Cos()
        self.mean = P.ReduceMean(keep_dims=False)
        self.sum = P.ReduceSum()
        self.eps = 1e-5
        self.dice = DiceCoefficient()

    def construct(
            self,
            y_true_cls,
            y_pred_cls,
            y_true_geo,
            y_pred_geo,
            training_mask):
        ans = self.sum(y_true_cls)
        classification_loss = self.dice(
            y_true_cls, y_pred_cls * (1 - training_mask))

        # n * 5 * h * w
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = self.split(y_true_geo)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = self.split(y_pred_geo)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = self.min(d2_gt, d2_pred) + self.min(d4_gt, d4_pred)
        h_union = self.min(d1_gt, d1_pred) + self.min(d3_gt, d3_pred)

        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        iou_loss_map = -self.log((area_intersect + 1.0) /
                                 (area_union + 1.0))  # iou_loss_map
        angle_loss_map = 1 - self.cos(theta_pred - theta_gt)  # angle_loss_map

        angle_loss = self.sum(angle_loss_map * y_true_cls) / ans
        iou_loss = self.sum(iou_loss_map * y_true_cls) / ans
        geo_loss = 10 * angle_loss + iou_loss

        return geo_loss + classification_loss


class EastWithLossCell(nn.Cell):
    def __init__(self, network):
        super(EastWithLossCell, self).__init__()
        self.east_network = network
        self.loss = EastLossBlock()

    def construct(self, img, true_cls, true_geo, training_mask):
        socre, geometry = self.east_network(img)
        loss = self.loss(
            true_cls,
            socre,
            true_geo,
            geometry,
            training_mask)
        return loss

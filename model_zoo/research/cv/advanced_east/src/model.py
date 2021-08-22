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
dataset processing.
"""
import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import ResizeNearestNeighbor
from mindspore import Tensor, ParameterTuple, Parameter
from mindspore.common.initializer import initializer, TruncatedNormal
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import numpy as np

from src.vgg import Vgg
from src.config import config as cfg


vgg_cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(num_classes=1000, args=None, phase="train"):
    """
    Get Vgg16 neural network with batch normalization.

    Args:
        num_classes (int): Class numbers. Default: 1000.
        args(namespace): param for net init.
        phase(str): train or test mode.

    Returns:
        Cell, cell instance of Vgg16 neural network with batch normalization.

    Examples:
        >>> vgg16(num_classes=1000, args=args)
    """

    if args is None:
        from src.config import cifar_cfg
        args = cifar_cfg
    net = Vgg(vgg_cfg['16'], num_classes=num_classes, args=args, batch_norm=args.batch_norm, phase=phase)
    return net

class AdvancedEast(nn.Cell):
    """
    East model
    Args:
            args
    """

    def __init__(self, args):
        super(AdvancedEast, self).__init__()
        self.device_target = args.device_target
        if self.device_target == 'GPU':

            self.vgg16 = vgg16()
            if args.is_train:
                param_dict = load_checkpoint(cfg.vgg_weights)
                load_param_into_net(self.vgg16, param_dict)

            self.bn1 = nn.BatchNorm2d(1024, momentum=0.99, eps=1e-3)
            self.conv1 = nn.Conv2d(1024, 128, 1, weight_init='XavierUniform', has_bias=True)
            self.relu1 = nn.ReLU()

            self.bn2 = nn.BatchNorm2d(128, momentum=0.99, eps=1e-3)
            self.conv2 = nn.Conv2d(128, 128, 3, padding=1, pad_mode='pad', weight_init='XavierUniform')
            self.relu2 = nn.ReLU()

            self.bn3 = nn.BatchNorm2d(384, momentum=0.99, eps=1e-3)
            self.conv3 = nn.Conv2d(384, 64, 1, weight_init='XavierUniform', has_bias=True)
            self.relu3 = nn.ReLU()

            self.bn4 = nn.BatchNorm2d(64, momentum=0.99, eps=1e-3)
            self.conv4 = nn.Conv2d(64, 64, 3, padding=1, pad_mode='pad', weight_init='XavierUniform')
            self.relu4 = nn.ReLU()

            self.bn5 = nn.BatchNorm2d(192, momentum=0.99, eps=1e-3)
            self.conv5 = nn.Conv2d(192, 32, 1, weight_init='XavierUniform', has_bias=True)
            self.relu5 = nn.ReLU()

            self.bn6 = nn.BatchNorm2d(32, momentum=0.99, eps=1e-3)
            self.conv6 = nn.Conv2d(32, 32, 3, padding=1, pad_mode='pad', weight_init='XavierUniform', has_bias=True)
            self.relu6 = nn.ReLU()

            self.bn7 = nn.BatchNorm2d(32, momentum=0.99, eps=1e-3)
            self.conv7 = nn.Conv2d(32, 32, 3, padding=1, pad_mode='pad', weight_init='XavierUniform', has_bias=True)
            self.relu7 = nn.ReLU()

            self.cat = P.Concat(axis=1)

            self.conv8 = nn.Conv2d(32, 1, 1, weight_init='XavierUniform', has_bias=True)
            self.conv9 = nn.Conv2d(32, 2, 1, weight_init='XavierUniform', has_bias=True)
            self.conv10 = nn.Conv2d(32, 4, 1, weight_init='XavierUniform', has_bias=True)
        else:
            if args.is_train:
                vgg_dict = np.load(cfg.vgg_npy, encoding='latin1', allow_pickle=True).item()
            shape_dict = {
                'conv1_1': [64, 3, 3, 3],
                'conv1_2': [64, 64, 3, 3],
                'conv2_1': [128, 64, 3, 3],
                'conv2_2': [128, 128, 3, 3],
                'conv3_1': [256, 128, 3, 3],
                'conv3_2': [256, 256, 3, 3],
                'conv3_3': [256, 256, 3, 3],
                'conv4_1': [512, 256, 3, 3],
                'conv4_2': [512, 512, 3, 3],
                'conv4_3': [512, 512, 3, 3],
                'conv5_1': [512, 512, 3, 3],
                'conv5_2': [512, 512, 3, 3],
                'conv5_3': [512, 512, 3, 3],
            }

            def get_var(name, idx):
                value = vgg_dict[name][idx]
                if idx == 0:
                    value = np.transpose(value, [3, 2, 0, 1])
                var = Tensor(value)
                return var

            def get_conv_var(name):
                filters = get_var(name, 0)
                biases = get_var(name, 1)
                return filters, biases

            class VGG_Conv(nn.Cell):
                """
                VGG16 network definition.
                """

                def __init__(self, name):
                    super(VGG_Conv, self).__init__()
                    if args.is_train:
                        filters, conv_biases = get_conv_var(name)
                        out_channels, in_channels, filter_size, _ = filters.shape
                    else:
                        out_channels, in_channels, filter_size, _ = shape_dict[name]
                    self.conv2d = P.Conv2D(out_channels, filter_size, pad_mode='same', mode=1)
                    self.bias_add = P.BiasAdd()
                    self.weight = Parameter(initializer(filters if args.is_train else TruncatedNormal(),
                                                        [out_channels, in_channels, filter_size, filter_size]),
                                            name='weight')
                    self.bias = Parameter(initializer(conv_biases if args.is_train else TruncatedNormal(),
                                                      [out_channels]), name='bias')
                    self.relu = P.ReLU()
                    self.gn = nn.GroupNorm(32, out_channels)

                def construct(self, x):
                    output = self.conv2d(x, self.weight)
                    output = self.bias_add(output, self.bias)
                    output = self.gn(output)
                    output = self.relu(output)
                    return output

            self.conv1_1 = VGG_Conv('conv1_1')
            self.conv1_2 = VGG_Conv('conv1_2')
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2_1 = VGG_Conv('conv2_1')
            self.conv2_2 = VGG_Conv('conv2_2')
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv3_1 = VGG_Conv('conv3_1')
            self.conv3_2 = VGG_Conv('conv3_2')
            self.conv3_3 = VGG_Conv('conv3_3')
            self.pool3 = nn.MaxPool2d(2, 2)
            self.conv4_1 = VGG_Conv('conv4_1')
            self.conv4_2 = VGG_Conv('conv4_2')
            self.conv4_3 = VGG_Conv('conv4_3')
            self.pool4 = nn.MaxPool2d(2, 2)
            self.conv5_1 = VGG_Conv('conv5_1')
            self.conv5_2 = VGG_Conv('conv5_2')
            self.conv5_3 = VGG_Conv('conv5_3')
            self.pool5 = nn.MaxPool2d(2, 2)
            self.merging1 = self.merging(i=2)
            self.merging2 = self.merging(i=3)
            self.merging3 = self.merging(i=4)
            self.last_bn = nn.GroupNorm(16, 32)
            self.conv_last = nn.Conv2d(32, 32, kernel_size=3, stride=1, has_bias=True, weight_init='XavierUniform')
            self.inside_score_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, has_bias=True,
                                               weight_init='XavierUniform')
            self.side_v_angle_conv = nn.Conv2d(32, 2, kernel_size=1, stride=1, has_bias=True,
                                               weight_init='XavierUniform')
            self.side_v_coord_conv = nn.Conv2d(32, 4, kernel_size=1, stride=1, has_bias=True,
                                               weight_init='XavierUniform')
            self.op_concat = P.Concat(axis=1)
            self.relu = P.ReLU()

    def merging(self, i=2):
        """
        def  merge layer
        """
        in_size = {'2': 1024, '3': 384, '4': 192}
        layers = [
            nn.Conv2d(in_size[str(i)], 128 // 2 ** (i - 2), kernel_size=1, stride=1, has_bias=True,
                      weight_init='XavierUniform'),
            nn.GroupNorm(16, 128 // 2 ** (i - 2)),
            nn.ReLU(),
            nn.Conv2d(128 // 2 ** (i - 2), 128 // 2 ** (i - 2), kernel_size=3, stride=1, has_bias=True,
                      weight_init='XavierUniform'),
            nn.GroupNorm(16, 128 // 2 ** (i - 2)),
            nn.ReLU()]
        return nn.SequentialCell(layers)

    def construct(self, x):
        """
            forward func
        """
        if self.device_target == 'GPU':
            l2, l3, l4, l5 = self.vgg16(x)
            h = l5

            _, _, h_, w_ = P.Shape()(h)
            g = ResizeNearestNeighbor((h_ * 2, w_ * 2))(h)
            c = self.cat((g, l4))

            c = self.bn1(c)
            c = self.conv1(c)
            c = self.relu1(c)

            h = self.bn2(c)
            h = self.conv2(h)
            h = self.relu2(h)

            _, _, h_, w_ = P.Shape()(h)
            g = ResizeNearestNeighbor((h_ * 2, w_ * 2))(h)
            c = self.cat((g, l3))

            c = self.bn3(c)
            c = self.conv3(c)
            c = self.relu3(c)

            h = self.bn4(c)
            h = self.conv4(h)  # bs 64 w/8 h/8
            h = self.relu4(h)

            _, _, h_, w_ = P.Shape()(h)
            g = ResizeNearestNeighbor((h_ * 2, w_ * 2))(h)
            c = self.cat((g, l2))

            c = self.bn5(c)
            c = self.conv5(c)
            c = self.relu5(c)

            h = self.bn6(c)
            h = self.conv6(h)  # bs 32 w/4 h/4
            h = self.relu6(h)

            g = self.bn7(h)
            g = self.conv7(g)  # bs 32 w/4 h/4
            g = self.relu7(g)
            # get output

            inside_score = self.conv8(g)
            side_v_code = self.conv9(g)
            side_v_coord = self.conv10(g)
            east_detect = self.cat((inside_score, side_v_code, side_v_coord))
        else:
            f4 = self.conv1_1(x)
            f4 = self.conv1_2(f4)
            f4 = self.pool1(f4)
            f4 = self.conv2_1(f4)
            f4 = self.conv2_2(f4)
            f4 = self.pool2(f4)
            f3 = self.conv3_1(f4)
            f3 = self.conv3_2(f3)
            f3 = self.conv3_3(f3)
            f3 = self.pool3(f3)
            f2 = self.conv4_1(f3)
            f2 = self.conv4_2(f2)
            f2 = self.conv4_3(f2)
            f2 = self.pool4(f2)
            f1 = self.conv5_1(f2)
            f1 = self.conv5_2(f1)
            f1 = self.conv5_3(f1)
            f1 = self.pool5(f1)
            h1 = f1
            _, _, h_, w_ = P.Shape()(h1)
            H1 = P.ResizeNearestNeighbor((h_ * 2, w_ * 2))(h1)
            concat1 = self.op_concat((H1, f2))
            h2 = self.merging1(concat1)
            _, _, h_, w_ = P.Shape()(h2)
            H2 = P.ResizeNearestNeighbor((h_ * 2, w_ * 2))(h2)
            concat2 = self.op_concat((H2, f3))
            h3 = self.merging2(concat2)
            _, _, h_, w_ = P.Shape()(h3)
            H3 = P.ResizeNearestNeighbor((h_ * 2, w_ * 2))(h3)
            concat3 = self.op_concat((H3, f4))
            h4 = self.merging3(concat3)
            before_output = self.relu(self.last_bn(self.conv_last(h4)))
            inside_score = self.inside_score_conv(before_output)
            side_v_angle = self.side_v_angle_conv(before_output)
            side_v_coord = self.side_v_coord_conv(before_output)
            east_detect = self.op_concat((inside_score, side_v_coord, side_v_angle))

        return east_detect



class EastWithLossCell(nn.Cell):
    """
    loss
    """

    def __init__(self, network):
        super(EastWithLossCell, self).__init__()
        self.East_network = network
        self.cat = P.Concat(axis=1)

    def dice_loss(self, gt_score, pred_score):
        """dice_loss1"""
        inter = P.ReduceSum()(gt_score * pred_score)
        union = P.ReduceSum()(gt_score) + P.ReduceSum()(pred_score) + 1e-5
        return 1. - (2 * (inter / union))

    def dice_loss2(self, gt_score, pred_score, mask):
        """dice_loss2"""
        inter = P.ReduceSum()(gt_score * pred_score * mask)
        union = P.ReduceSum()(gt_score * mask) + P.ReduceSum()(pred_score * mask) + 1e-5
        return 1. - (2 * (inter / union))

    def quad_loss(self, y_true, y_pred,
                  lambda_inside_score_loss=0.2,
                  lambda_side_vertex_code_loss=0.1,
                  lambda_side_vertex_coord_loss=1.0,
                  epsilon=1e-4):
        """quad loss"""
        y_true = P.Transpose()(y_true, (0, 2, 3, 1))
        y_pred = P.Transpose()(y_pred, (0, 2, 3, 1))
        logits = y_pred[:, :, :, :1]
        labels = y_true[:, :, :, :1]
        predicts = P.Sigmoid()(logits)
        inside_score_loss = self.dice_loss(labels, predicts)
        inside_score_loss = inside_score_loss * lambda_inside_score_loss
        # loss for side_vertex_code
        vertex_logitsp = P.Sigmoid()(y_pred[:, :, :, 1:2])
        vertex_labelsp = y_true[:, :, :, 1:2]
        vertex_logitsn = P.Sigmoid()(y_pred[:, :, :, 2:3])
        vertex_labelsn = y_true[:, :, :, 2:3]
        labels2 = y_true[:, :, :, 1:2]
        side_vertex_code_lossp = self.dice_loss2(vertex_labelsp, vertex_logitsp, labels)
        side_vertex_code_lossn = self.dice_loss2(vertex_labelsn, vertex_logitsn, labels2)
        side_vertex_code_loss = (side_vertex_code_lossp + side_vertex_code_lossn) * lambda_side_vertex_code_loss
        # loss for side_vertex_coord delta
        g_hat = y_pred[:, :, :, 3:]  # N*W*H*8
        g_true = y_true[:, :, :, 3:]
        vertex_weights = P.Cast()(P.Equal()(y_true[:, :, :, 1], 1), mindspore.float32)

        pixel_wise_smooth_l1norm = self.smooth_l1_loss(g_hat, g_true, vertex_weights)
        side_vertex_coord_loss = P.ReduceSum()(pixel_wise_smooth_l1norm) / (
            P.ReduceSum()(vertex_weights) + epsilon)
        side_vertex_coord_loss = side_vertex_coord_loss * lambda_side_vertex_coord_loss
        return inside_score_loss + side_vertex_code_loss + side_vertex_coord_loss

    def smooth_l1_loss(self, prediction_tensor, target_tensor, weights):
        """smooth l1 loss"""
        n_q = P.Reshape()(self.quad_norm(target_tensor), weights.shape)
        diff = P.SmoothL1Loss()(prediction_tensor, target_tensor)
        pixel_wise_smooth_l1norm = P.ReduceSum()(diff, -1) / n_q * weights
        return pixel_wise_smooth_l1norm

    def quad_norm(self, g_true, epsilon=1e-4):
        """ quad norm"""
        shape = g_true.shape
        delta_xy_matrix = P.Reshape()(g_true, (shape[0] * shape[1] * shape[2], 2, 2))
        diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
        square = diff * diff
        distance = P.Sqrt()(P.ReduceSum()(square, -1))
        distance = distance * 4.0
        distance = distance + epsilon
        return P.Reshape()(distance, (shape[0], shape[1], shape[2]))

    def construct(self, image, label):
        y_pred = self.East_network(image)
        loss = self.quad_loss(label, y_pred)
        return loss


class TrainStepWrap(nn.Cell):
    """
    train net
    """

    def __init__(self, network):
        super(TrainStepWrap, self).__init__()
        self.network = network
        self.network.set_train()
        self.weights = ParameterTuple(network.trainable_params())
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = 1.0

    def construct(self, image, label):
        weights = self.weights
        loss = self.network(image, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(image, label, sens)
        self.optimizer(grads)
        return loss


def get_AdvancedEast_net(args):
    """
    Get network of wide&deep model.
    """
    AdvancedEast_net = AdvancedEast(args)
    loss_net = EastWithLossCell(AdvancedEast_net)
    train_net = TrainStepWrap(loss_net)
    return loss_net, train_net

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

"""YOLOv3 based on ResNet18."""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C


def weight_variable():
    """Weight variable."""
    return TruncatedNormal(0.02)


class _conv2d(nn.Cell):
    """Create Conv2D with padding."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(_conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=0, pad_mode='same',
                              weight_init=weight_variable())
    def construct(self, x):
        x = self.conv(x)
        return x


def _fused_bn(channels, momentum=0.99):
    """Get a fused batchnorm."""
    return nn.BatchNorm2d(channels, momentum=momentum)


def _conv_bn_relu(in_channel,
                  out_channel,
                  ksize,
                  stride=1,
                  padding=0,
                  dilation=1,
                  alpha=0.1,
                  momentum=0.99,
                  pad_mode="same"):
    """Get a conv2d batchnorm and relu layer."""
    return nn.SequentialCell(
        [nn.Conv2d(in_channel,
                   out_channel,
                   kernel_size=ksize,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channel, momentum=momentum),
         nn.LeakyReLU(alpha)]
    )


class BasicBlock(nn.Cell):
    """
    ResNet basic block.

    Args:
        in_channels (int): Input channel.
        out_channels (int): Output channel.
        stride (int): Stride size for the initial convolutional layer. Default:1.
        momentum (float): Momentum for batchnorm layer. Default:0.1.

    Returns:
        Tensor, output tensor.

    Examples:
        BasicBlock(3,256,stride=2,down_sample=True).
    """
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 momentum=0.99):
        super(BasicBlock, self).__init__()

        self.conv1 = _conv2d(in_channels, out_channels, 3, stride=stride)
        self.bn1 = _fused_bn(out_channels, momentum=momentum)
        self.conv2 = _conv2d(out_channels, out_channels, 3)
        self.bn2 = _fused_bn(out_channels, momentum=momentum)
        self.relu = P.ReLU()
        self.down_sample_layer = None
        self.downsample = (in_channels != out_channels)
        if self.downsample:
            self.down_sample_layer = _conv2d(in_channels, out_channels, 1, stride=stride)
        self.add = P.Add()

    def construct(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.down_sample_layer(identity)

        out = self.add(x, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet network.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of different layers.
        in_channels (int): Input channel.
        out_channels (int): Output channel.
        num_classes (int): Class number. Default:100.

    Returns:
        Tensor, output tensor.

    Examples:
        ResNet(ResidualBlock,
               [3, 4, 6, 3],
               [64, 256, 512, 1024],
               [256, 512, 1024, 2048],
               100).
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides=None,
                 num_classes=80):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of "
                             "layer_num, inchannel, outchannel list must be 4!")

        self.conv1 = _conv2d(3, 64, 7, stride=2)
        self.bn1 = _fused_bn(64)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.num_classes = num_classes
        if num_classes:
            self.reduce_mean = P.ReduceMean(keep_dims=True)
            self.end_point = nn.Dense(out_channels[3], num_classes, has_bias=True,
                                      weight_init=weight_variable(),
                                      bias_init=weight_variable())
            self.squeeze = P.Squeeze(axis=(2, 3))

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Make Layer for ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the initial convolutional layer.

        Returns:
            SequentialCell, the output layer.

        Examples:
            _make_layer(BasicBlock, 3, 128, 256, 2).
        """
        layers = []

        resblk = block(in_channel, out_channel, stride=stride)
        layers.append(resblk)

        for _ in range(1, layer_num - 1):
            resblk = block(out_channel, out_channel, stride=1)
            layers.append(resblk)

        resblk = block(out_channel, out_channel, stride=1)
        layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = c5
        if self.num_classes:
            out = self.reduce_mean(c5, (2, 3))
            out = self.squeeze(out)
            out = self.end_point(out)

        return c3, c4, out


def resnet18(class_num=10):
    """
    Get ResNet18 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet18 neural network.

    Examples:
        resnet18(100).
    """
    return ResNet(BasicBlock,
                  [2, 2, 2, 2],
                  [64, 64, 128, 256],
                  [64, 128, 256, 512],
                  [1, 2, 2, 2],
                  num_classes=class_num)


class YoloBlock(nn.Cell):
    """
    YoloBlock for YOLOv3.

    Args:
        in_channels (int): Input channel.
        out_chls (int): Middle channel.
        out_channels (int): Output channel.

    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3).

    Examples:
        YoloBlock(1024, 512, 255).

    """
    def __init__(self, in_channels, out_chls, out_channels):
        super(YoloBlock, self).__init__()
        out_chls_2 = out_chls * 2

        self.conv0 = _conv_bn_relu(in_channels, out_chls, ksize=1)
        self.conv1 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv2 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv3 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv4 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv5 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv6 = nn.Conv2d(out_chls_2, out_channels, kernel_size=1, stride=1, has_bias=True)

    def construct(self, x):
        c1 = self.conv0(x)
        c2 = self.conv1(c1)

        c3 = self.conv2(c2)
        c4 = self.conv3(c3)

        c5 = self.conv4(c4)
        c6 = self.conv5(c5)

        out = self.conv6(c6)
        return c5, out


class YOLOv3(nn.Cell):
    """
     YOLOv3 Network.

     Note:
         backbone = resnet18.

     Args:
         feature_shape (list): Input image shape, [N,C,H,W].
         backbone_shape (list): resnet18 output channels shape.
         backbone (Cell): Backbone Network.
         out_channel (int): Output channel.

     Returns:
         Tensor, output tensor.

     Examples:
         YOLOv3(feature_shape=[1,3,416,416],
                backbone_shape=[64, 128, 256, 512, 1024]
                backbone=darknet53(),
                out_channel=255).
     """
    def __init__(self, feature_shape, backbone_shape, backbone, out_channel):
        super(YOLOv3, self).__init__()
        self.out_channel = out_channel
        self.net = backbone
        self.backblock0 = YoloBlock(backbone_shape[-1], out_chls=backbone_shape[-2], out_channels=out_channel)

        self.conv1 = _conv_bn_relu(in_channel=backbone_shape[-2], out_channel=backbone_shape[-2]//2, ksize=1)
        self.upsample1 = P.ResizeNearestNeighbor((feature_shape[2]//16, feature_shape[3]//16))
        self.backblock1 = YoloBlock(in_channels=backbone_shape[-2]+backbone_shape[-3],
                                    out_chls=backbone_shape[-3],
                                    out_channels=out_channel)

        self.conv2 = _conv_bn_relu(in_channel=backbone_shape[-3], out_channel=backbone_shape[-3]//2, ksize=1)
        self.upsample2 = P.ResizeNearestNeighbor((feature_shape[2]//8, feature_shape[3]//8))
        self.backblock2 = YoloBlock(in_channels=backbone_shape[-3]+backbone_shape[-4],
                                    out_chls=backbone_shape[-4],
                                    out_channels=out_channel)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        # input_shape of x is (batch_size, 3, h, w)
        # feature_map1 is (batch_size, backbone_shape[2], h/8, w/8)
        # feature_map2 is (batch_size, backbone_shape[3], h/16, w/16)
        # feature_map3 is (batch_size, backbone_shape[4], h/32, w/32)
        feature_map1, feature_map2, feature_map3 = self.net(x)
        con1, big_object_output = self.backblock0(feature_map3)

        con1 = self.conv1(con1)
        ups1 = self.upsample1(con1)
        con1 = self.concat((ups1, feature_map2))
        con2, medium_object_output = self.backblock1(con1)

        con2 = self.conv2(con2)
        ups2 = self.upsample2(con2)
        con3 = self.concat((ups2, feature_map1))
        _, small_object_output = self.backblock2(con3)

        return big_object_output, medium_object_output, small_object_output


class DetectionBlock(nn.Cell):
    """
     YOLOv3 detection Network. It will finally output the detection result.

     Args:
         scale (str): Character, scale.
         config (Class): YOLOv3 config.

     Returns:
         Tuple, tuple of output tensor,(f1,f2,f3).

     Examples:
         DetectionBlock(scale='l',stride=32).
     """

    def __init__(self, scale, config):
        super(DetectionBlock, self).__init__()

        self.config = config
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4 + 1 + self.config.num_classes
        self.ignore_threshold = 0.5
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.reshape = P.Reshape()
        self.tile = P.Tile()
        self.concat = P.Concat(axis=-1)
        self.input_shape = Tensor(tuple(config.img_shape[::-1]), ms.float32)

    def construct(self, x):
        num_batch = P.Shape()(x)[0]
        grid_size = P.Shape()(x)[2:4]

        # Reshape and transpose the feature to [n, 3, grid_size[0], grid_size[1], num_attrib]
        prediction = P.Reshape()(x, (num_batch,
                                     self.num_anchors_per_scale,
                                     self.num_attrib,
                                     grid_size[0],
                                     grid_size[1]))
        prediction = P.Transpose()(prediction, (0, 3, 4, 1, 2))

        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        grid_x = P.Cast()(F.tuple_to_array(range_x), ms.float32)
        grid_y = P.Cast()(F.tuple_to_array(range_y), ms.float32)
        # Tensor of shape [grid_size[0], grid_size[1], 1, 1] representing the coordinate of x/y axis for each grid
        grid_x = self.tile(self.reshape(grid_x, (1, 1, -1, 1, 1)), (1, grid_size[0], 1, 1, 1))
        grid_y = self.tile(self.reshape(grid_y, (1, -1, 1, 1, 1)), (1, 1, grid_size[1], 1, 1))
        # Shape is [grid_size[0], grid_size[1], 1, 2]
        grid = self.concat((grid_x, grid_y))

        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]
        box_confidence = prediction[:, :, :, :, 4:5]
        box_probs = prediction[:, :, :, :, 5:]

        box_xy = (self.sigmoid(box_xy) + grid) / P.Cast()(F.tuple_to_array((grid_size[1], grid_size[0])), ms.float32)
        box_wh = P.Exp()(box_wh) * self.anchors / self.input_shape
        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)

        if self.training:
            return grid, prediction, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_probs


class Iou(nn.Cell):
    """Calculate the iou of boxes."""
    def __init__(self):
        super(Iou, self).__init__()
        self.min = P.Minimum()
        self.max = P.Maximum()

    def construct(self, box1, box2):
        box1_xy = box1[:, :, :, :, :, :2]
        box1_wh = box1[:, :, :, :, :, 2:4]
        box1_mins = box1_xy - box1_wh / F.scalar_to_array(2.0)
        box1_maxs = box1_xy + box1_wh / F.scalar_to_array(2.0)

        box2_xy = box2[:, :, :, :, :, :2]
        box2_wh = box2[:, :, :, :, :, 2:4]
        box2_mins = box2_xy - box2_wh / F.scalar_to_array(2.0)
        box2_maxs = box2_xy + box2_wh / F.scalar_to_array(2.0)

        intersect_mins = self.max(box1_mins, box2_mins)
        intersect_maxs = self.min(box1_maxs, box2_maxs)
        intersect_wh = self.max(intersect_maxs - intersect_mins, F.scalar_to_array(0.0))

        intersect_area = P.Squeeze(-1)(intersect_wh[:, :, :, :, :, 0:1]) * \
                         P.Squeeze(-1)(intersect_wh[:, :, :, :, :, 1:2])
        box1_area = P.Squeeze(-1)(box1_wh[:, :, :, :, :, 0:1]) * P.Squeeze(-1)(box1_wh[:, :, :, :, :, 1:2])
        box2_area = P.Squeeze(-1)(box2_wh[:, :, :, :, :, 0:1]) * P.Squeeze(-1)(box2_wh[:, :, :, :, :, 1:2])

        iou = intersect_area / (box1_area + box2_area - intersect_area)
        return iou


class YoloLossBlock(nn.Cell):
    """
     YOLOv3 Loss block cell. It will finally output loss of the scale.

     Args:
         scale (str): Three scale here, 's', 'm' and 'l'.
         config (Class): The default config of YOLOv3.

     Returns:
         Tensor, loss of the scale.

     Examples:
         YoloLossBlock('l', ConfigYOLOV3ResNet18()).
     """

    def __init__(self, scale, config):
        super(YoloLossBlock, self).__init__()
        self.config = config
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.ignore_threshold = Tensor(self.config.ignore_threshold, ms.float32)
        self.concat = P.Concat(axis=-1)
        self.iou = Iou()
        self.cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = P.ReduceSum()
        self.reduce_max = P.ReduceMax(keep_dims=False)
        self.input_shape = Tensor(tuple(config.img_shape[::-1]), ms.float32)

    def construct(self, grid, prediction, pred_xy, pred_wh, y_true, gt_box):

        object_mask = y_true[:, :, :, :, 4:5]
        class_probs = y_true[:, :, :, :, 5:]

        grid_shape = P.Shape()(prediction)[1:3]
        grid_shape = P.Cast()(F.tuple_to_array(grid_shape[::-1]), ms.float32)

        pred_boxes = self.concat((pred_xy, pred_wh))
        true_xy = y_true[:, :, :, :, :2] * grid_shape - grid
        true_wh = y_true[:, :, :, :, 2:4]
        true_wh = P.Select()(P.Equal()(true_wh, 0.0),
                             P.Fill()(P.DType()(true_wh), P.Shape()(true_wh), 1.0),
                             true_wh)
        true_wh = P.Log()(true_wh / self.anchors * self.input_shape)
        box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]

        gt_shape = P.Shape()(gt_box)
        gt_box = P.Reshape()(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))

        iou = self.iou(P.ExpandDims()(pred_boxes, -2), gt_box) # [batch, grid[0], grid[1], num_anchor, num_gt]
        best_iou = self.reduce_max(iou, -1) # [batch, grid[0], grid[1], num_anchor]
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = P.Cast()(ignore_mask, ms.float32)
        ignore_mask = P.ExpandDims()(ignore_mask, -1)
        ignore_mask = F.stop_gradient(ignore_mask)

        xy_loss = object_mask * box_loss_scale * self.cross_entropy(prediction[:, :, :, :, :2], true_xy)
        wh_loss = object_mask * box_loss_scale * 0.5 * P.Square()(true_wh - prediction[:, :, :, :, 2:4])
        confidence_loss = self.cross_entropy(prediction[:, :, :, :, 4:5], object_mask)
        confidence_loss = object_mask * confidence_loss + (1 - object_mask) * confidence_loss * ignore_mask
        class_loss = object_mask * self.cross_entropy(prediction[:, :, :, :, 5:], class_probs)

        # Get smooth loss
        xy_loss = self.reduce_sum(xy_loss, ())
        wh_loss = self.reduce_sum(wh_loss, ())
        confidence_loss = self.reduce_sum(confidence_loss, ())
        class_loss = self.reduce_sum(class_loss, ())

        loss = xy_loss + wh_loss + confidence_loss + class_loss
        return loss / P.Shape()(prediction)[0]


class yolov3_resnet18(nn.Cell):
    """
    ResNet based YOLOv3 network.

    Args:
        config (Class): YOLOv3 config.

    Returns:
        Cell, cell instance of ResNet based YOLOv3 neural network.

    Examples:
        yolov3_resnet18(80, [1,3,416,416]).
    """

    def __init__(self, config):
        super(yolov3_resnet18, self).__init__()
        self.config = config

        # YOLOv3 network
        self.feature_map = YOLOv3(feature_shape=self.config.feature_shape,
                                  backbone=ResNet(BasicBlock,
                                                  self.config.backbone_layers,
                                                  self.config.backbone_input_shape,
                                                  self.config.backbone_shape,
                                                  self.config.backbone_stride,
                                                  num_classes=None),
                                  backbone_shape=self.config.backbone_shape,
                                  out_channel=self.config.out_channel)

        # prediction on the default anchor boxes
        self.detect_1 = DetectionBlock('l', self.config)
        self.detect_2 = DetectionBlock('m', self.config)
        self.detect_3 = DetectionBlock('s', self.config)

    def construct(self, x):
        big_object_output, medium_object_output, small_object_output = self.feature_map(x)
        output_big = self.detect_1(big_object_output)
        output_me = self.detect_2(medium_object_output)
        output_small = self.detect_3(small_object_output)

        return output_big, output_me, output_small


class YoloWithLossCell(nn.Cell):
    """"
    Provide YOLOv3 training loss through network.

    Args:
        network (Cell): The training network.
        config (Class): YOLOv3 config.

    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, network, config):
        super(YoloWithLossCell, self).__init__()
        self.yolo_network = network
        self.config = config
        self.loss_big = YoloLossBlock('l', self.config)
        self.loss_me = YoloLossBlock('m', self.config)
        self.loss_small = YoloLossBlock('s', self.config)

    def construct(self, x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
        yolo_out = self.yolo_network(x)
        loss_l = self.loss_big(yolo_out[0][0], yolo_out[0][1], yolo_out[0][2], yolo_out[0][3], y_true_0, gt_0)
        loss_m = self.loss_me(yolo_out[1][0], yolo_out[1][1], yolo_out[1][2], yolo_out[1][3], y_true_1, gt_1)
        loss_s = self.loss_small(yolo_out[2][0], yolo_out[2][1], yolo_out[2][2], yolo_out[2][3], y_true_2, gt_2)
        return loss_l + loss_m + loss_s


class TrainingWrapper(nn.Cell):
    """
    Encapsulation class of YOLOv3 network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))


class YoloBoxScores(nn.Cell):
    """
    Calculate the boxes of the original picture size and the score of each box.

    Args:
        config (Class): YOLOv3 config.

    Returns:
        Tensor, the boxes of the original picture size.
        Tensor, the score of each box.
    """
    def __init__(self, config):
        super(YoloBoxScores, self).__init__()
        self.input_shape = Tensor(np.array(config.img_shape), ms.float32)
        self.num_classes = config.num_classes

    def construct(self, box_xy, box_wh, box_confidence, box_probs, image_shape):
        batch_size = F.shape(box_xy)[0]
        x = box_xy[:, :, :, :, 0:1]
        y = box_xy[:, :, :, :, 1:2]
        box_yx = P.Concat(-1)((y, x))
        w = box_wh[:, :, :, :, 0:1]
        h = box_wh[:, :, :, :, 1:2]
        box_hw = P.Concat(-1)((h, w))

        new_shape = P.Round()(image_shape * P.ReduceMin()(self.input_shape / image_shape))
        offset = (self.input_shape - new_shape) / 2.0 / self.input_shape
        scale = self.input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw = box_hw * scale

        box_min = box_yx - box_hw / 2.0
        box_max = box_yx + box_hw / 2.0
        boxes = P.Concat(-1)((box_min[:, :, :, :, 0:1],
                              box_min[:, :, :, :, 1:2],
                              box_max[:, :, :, :, 0:1],
                              box_max[:, :, :, :, 1:2]))
        image_scale = P.Tile()(image_shape, (1, 2))
        boxes = boxes * image_scale
        boxes = F.reshape(boxes, (batch_size, -1, 4))
        boxes_scores = box_confidence * box_probs
        boxes_scores = F.reshape(boxes_scores, (batch_size, -1, self.num_classes))
        return boxes, boxes_scores


class YoloWithEval(nn.Cell):
    """
    Encapsulation class of YOLOv3 evaluation.

    Args:
        network (Cell): The training network. Note that loss function and optimizer must not be added.
        config (Class): YOLOv3 config.

    Returns:
        Tensor, the boxes of the original picture size.
        Tensor, the score of each box.
        Tensor, the original picture size.
    """
    def __init__(self, network, config):
        super(YoloWithEval, self).__init__()
        self.yolo_network = network
        self.box_score_0 = YoloBoxScores(config)
        self.box_score_1 = YoloBoxScores(config)
        self.box_score_2 = YoloBoxScores(config)

    def construct(self, x, image_shape):
        yolo_output = self.yolo_network(x)
        boxes_0, boxes_scores_0 = self.box_score_0(*yolo_output[0], image_shape)
        boxes_1, boxes_scores_1 = self.box_score_1(*yolo_output[1], image_shape)
        boxes_2, boxes_scores_2 = self.box_score_2(*yolo_output[2], image_shape)
        boxes = P.Concat(1)((boxes_0, boxes_1, boxes_2))
        boxes_scores = P.Concat(1)((boxes_scores_0, boxes_scores_1, boxes_scores_2))
        return boxes, boxes_scores, image_shape

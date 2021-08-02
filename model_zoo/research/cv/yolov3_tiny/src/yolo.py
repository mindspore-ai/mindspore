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
"""YOLO based on DarkNet."""
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C

from src.tiny import Tiny
from src.loss import ConfidenceLoss, ClassLoss
from model_utils.config import config

class YOLOv3_Tiny(nn.Cell):
    """
     YOLO Network.

     Args:
         backbone: Cell. Backbone Network.

     Returns:
         Tensor, output tensor.

     Examples:
         YOLOv3_Tiny(backbone)
     """
    def __init__(self, backbone):
        super(YOLOv3_Tiny, self).__init__()

        self.backbone = backbone
        self.head1 = nn.Conv2d(512, 255, kernel_size=1, stride=1, has_bias=True)
        self.head2 = nn.Conv2d(256, 255, kernel_size=1, stride=1, has_bias=True)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        # input_shape of x is (batch_size, 3, h, w)
        # feature_map1 is (batch_size, backbone_shape1, h/32, w/32)
        # feature_map2 is (batch_size, backbone_shape2, h/16, w/16)

        feature_map1, feature_map2 = self.backbone(x)
        big_object_output = self.head1(feature_map1)
        small_object_output = self.head2(feature_map2)

        return big_object_output, small_object_output



class DetectionBlock(nn.Cell):
    """
     YOLO detection Network. It will finally output the detection result.

     Args:
         scale: Character.
         config: Configuration.
         is_training: Bool, Whether train or not, default True.

     Returns:
         Tuple, tuple of output tensor,(f1,f2,f3).

     Examples:
         DetectionBlock(scale='l',stride=32)
     """

    def __init__(self, scale, config_db=config, is_training=True):
        super(DetectionBlock, self).__init__()
        self.config = config_db
        if scale == 's':
            idx = (0, 1, 2)
            self.scale_x_y = 1.2
            self.offset_x_y = 0.1
        elif scale == 'l':
            idx = (3, 4, 5)
            self.scale_x_y = 1.1
            self.offset_x_y = 0.05
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4+1+self.config.num_classes
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.reshape = P.Reshape()
        self.tile = P.Tile()
        self.concat = P.Concat(axis=-1)
        self.conf_training = is_training

    def construct(self, x, input_shape):
        """construct method"""
        num_batch = P.Shape()(x)[0]
        grid_size = P.Shape()(x)[2:4]

        # Reshape and transpose the feature to [n, grid_size[0], grid_size[1], 3, num_attrib]
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
        # [batch, gridx, gridy, 1, 1]
        grid_x = self.tile(self.reshape(grid_x, (1, 1, -1, 1, 1)), (1, grid_size[0], 1, 1, 1))
        grid_y = self.tile(self.reshape(grid_y, (1, -1, 1, 1, 1)), (1, 1, grid_size[1], 1, 1))
        # Shape is [grid_size[0], grid_size[1], 1, 2]
        grid = self.concat((grid_x, grid_y))

        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]
        box_confidence = prediction[:, :, :, :, 4:5]
        box_probs = prediction[:, :, :, :, 5:]

        # gridsize1 is x
        # gridsize0 is y
        box_xy = (self.scale_x_y * self.sigmoid(box_xy) - self.offset_x_y + grid) / \
                 P.Cast()(F.tuple_to_array((grid_size[1], grid_size[0])), ms.float32)
        # box_wh is w->h
        box_wh = P.Exp()(box_wh) * self.anchors / input_shape
        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)

        if self.conf_training:
            return prediction, box_xy, box_wh
        return self.concat((box_xy, box_wh, box_confidence, box_probs))


class Iou(nn.Cell):
    """Calculate the iou of boxes"""
    def __init__(self):
        super(Iou, self).__init__()
        self.min = P.Minimum()
        self.max = P.Maximum()

    def construct(self, box1, box2):
        """
        box1: pred_box [batch, gx, gy, anchors, 1,      4] ->4: [x_center, y_center, w, h]
        box2: gt_box   [batch, 1,  1,  1,       maxbox, 4]
        convert to topLeft and rightDown
        """
        box1_xy = box1[:, :, :, :, :, :2]
        box1_wh = box1[:, :, :, :, :, 2:4]
        box1_mins = box1_xy - box1_wh / F.scalar_to_array(2.0) # topLeft
        box1_maxs = box1_xy + box1_wh / F.scalar_to_array(2.0) # rightDown

        box2_xy = box2[:, :, :, :, :, :2]
        box2_wh = box2[:, :, :, :, :, 2:4]
        box2_mins = box2_xy - box2_wh / F.scalar_to_array(2.0)
        box2_maxs = box2_xy + box2_wh / F.scalar_to_array(2.0)

        intersect_mins = self.max(box1_mins, box2_mins)
        intersect_maxs = self.min(box1_maxs, box2_maxs)
        intersect_wh = self.max(intersect_maxs - intersect_mins, F.scalar_to_array(0.0))
        # P.squeeze: for effiecient slice
        intersect_area = P.Squeeze(-1)(intersect_wh[:, :, :, :, :, 0:1]) * \
                         P.Squeeze(-1)(intersect_wh[:, :, :, :, :, 1:2])
        box1_area = P.Squeeze(-1)(box1_wh[:, :, :, :, :, 0:1]) * P.Squeeze(-1)(box1_wh[:, :, :, :, :, 1:2])
        box2_area = P.Squeeze(-1)(box2_wh[:, :, :, :, :, 0:1]) * P.Squeeze(-1)(box2_wh[:, :, :, :, :, 1:2])
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        # iou : [batch, gx, gy, anchors, maxboxes]
        return iou


class YoloLossBlock(nn.Cell):
    """
    Loss block cell of YOLO network.
    """
    def __init__(self, scale, config_yb=None):
        super(YoloLossBlock, self).__init__()
        self.config = config_yb
        if scale == 's':
            # anchor mask
            idx = (0, 1, 2)
        elif scale == 'l':
            idx = (3, 4, 5)

        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.ignore_threshold = Tensor(self.config.ignore_threshold, ms.float32)
        self.concat = P.Concat(axis=-1)
        self.iou = Iou()
        self.reduce_max = P.ReduceMax(keep_dims=False)
        self.confidence_loss = ConfidenceLoss()
        self.class_loss = ClassLoss()
        self.reduce_sum = P.ReduceSum()
        self.giou = Giou()
        self.coord_scale = config.coord_scale

    def construct(self, prediction, pred_xy, pred_wh, y_true, gt_box, input_shape):
        """
        prediction : origin output from yolo
        pred_xy: (sigmoid(xy)+grid)/grid_size
        pred_wh: (exp(wh)*anchors)/input_shape
        y_true : after normalize
        gt_box: [batch, maxboxes, xyhw] after normalize
        """
        object_mask = y_true[:, :, :, :, 4:5]
        class_probs = y_true[:, :, :, :, 5:]
        true_boxes = y_true[:, :, :, :, :4]

        grid_shape = P.Shape()(prediction)[1:3]
        grid_shape = P.Cast()(F.tuple_to_array(grid_shape[::-1]), ms.float32)

        pred_boxes = self.concat((pred_xy, pred_wh))
        true_wh = y_true[:, :, :, :, 2:4]
        true_wh = P.Select()(P.Equal()(true_wh, 0.0),
                             P.Fill()(P.DType()(true_wh),
                                      P.Shape()(true_wh), 1.0),
                             true_wh)
        true_wh = P.Log()(true_wh / self.anchors * input_shape)
        # 2-w*h for large picture, use small scale, since small obj need more precise
        box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]

        gt_shape = P.Shape()(gt_box)
        gt_box = P.Reshape()(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))

        # add one more dimension for broadcast
        iou = self.iou(P.ExpandDims()(pred_boxes, -2), gt_box)
        # gt_box is x,y,h,w after normalize
        # [batch, grid[0], grid[1], num_anchor, num_gt]
        best_iou = self.reduce_max(iou, -1)
        # [batch, grid[0], grid[1], num_anchor]

        # ignore_mask IOU too small
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = P.Cast()(ignore_mask, ms.float32)
        ignore_mask = P.ExpandDims()(ignore_mask, -1)
        # ignore_mask backpro will cause a lot maximunGrad and minimumGrad time consume.
        # so we turn off its gradient
        ignore_mask = F.stop_gradient(ignore_mask)

        confidence_loss = self.confidence_loss(object_mask, prediction[:, :, :, :, 4:5], ignore_mask)
        class_loss = self.class_loss(object_mask, prediction[:, :, :, :, 5:], class_probs)

        object_mask_me = P.Reshape()(object_mask, (-1, 1))  # [8, 72, 72, 3, 1]
        box_loss_scale_me = P.Reshape()(box_loss_scale, (-1, 1))
        pred_boxes_me = xywh2x1y1x2y2(pred_boxes)
        pred_boxes_me = P.Reshape()(pred_boxes_me, (-1, 4))
        true_boxes_me = xywh2x1y1x2y2(true_boxes)
        true_boxes_me = P.Reshape()(true_boxes_me, (-1, 4))
        ciou = self.giou(pred_boxes_me, true_boxes_me)
        ciou_loss = object_mask_me * box_loss_scale_me * (1 - ciou)
        ciou_loss_me = self.reduce_sum(ciou_loss, ())
        loss = ciou_loss_me * self.coord_scale + confidence_loss + class_loss
        batch_size = P.Shape()(prediction)[0]
        return loss / batch_size


class YOLOV3(nn.Cell):
    """
    YOLO network.

    Args:
        is_training: Bool. Whether train or not.

    Returns:
        Cell, cell instance of YOLO neural network.

    Examples:
        YOLO(True)
    """

    def __init__(self, is_training, config_y=config):
        super(YOLOV3, self).__init__()
        self.config = config_y

        # YOLO network
        self.feature_map = YOLOv3_Tiny(backbone=Tiny())

        # prediction on the default anchor boxes
        self.detect_1 = DetectionBlock('l', is_training=is_training)
        self.detect_2 = DetectionBlock('s', is_training=is_training)

    def construct(self, x, input_shape):
        big_object_output, small_object_output = self.feature_map(x)
        output_big = self.detect_1(big_object_output, input_shape)
        output_small = self.detect_2(small_object_output, input_shape)
        # big is the final output which has smallest feature map
        return output_big, output_small


class YOLOV3_Infer(nn.Cell):
    """
    YOLOV3 Infer.
    """
    def __init__(self, inputshape):
        super(YOLOV3_Infer, self).__init__()
        self.network = YOLOV3(is_training=False)
        self.inputshape = inputshape

    def construct(self, x):
        return self.network(x, self.inputshape)


class YoloWithLossCell(nn.Cell):
    """YOLO loss."""
    def __init__(self, network, config_ywl=config):
        super(YoloWithLossCell, self).__init__()
        self.yolo_network = network
        self.config = config_ywl
        self.loss_big = YoloLossBlock('l', self.config)
        self.loss_small = YoloLossBlock('s', self.config)
        self.tenser_to_array = P.TupleToArray()
        self.large_scale = self.config.large_scale

    def construct(self, x, y_true_0, y_true_1, gt_0, gt_1):
        input_shape = F.shape(x)[2:4]
        input_shape = F.cast(self.tenser_to_array(input_shape), ms.float32)
        yolo_out = self.yolo_network(x, input_shape)
        loss_l = self.loss_big(*yolo_out[0], y_true_0, gt_0, input_shape)
        loss_s = self.loss_small(*yolo_out[1], y_true_1, gt_1, input_shape)
        return loss_l * self.large_scale + loss_s


class TrainingWrapper(nn.Cell):
    """Training wrapper."""
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
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
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))


class Giou(nn.Cell):
    """Calculating giou"""
    def __init__(self):
        super(Giou, self).__init__()
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.concat = P.Concat(axis=1)
        self.mean = P.ReduceMean()
        self.div = P.RealDiv()
        self.eps = 0.000001

    def construct(self, box_p, box_gt):
        """construct method"""
        box_p_area = (box_p[..., 2:3] - box_p[..., 0:1]) * (box_p[..., 3:4] - box_p[..., 1:2])
        box_gt_area = (box_gt[..., 2:3] - box_gt[..., 0:1]) * (box_gt[..., 3:4] - box_gt[..., 1:2])
        x_1 = self.max(box_p[..., 0:1], box_gt[..., 0:1])
        x_2 = self.min(box_p[..., 2:3], box_gt[..., 2:3])
        y_1 = self.max(box_p[..., 1:2], box_gt[..., 1:2])
        y_2 = self.min(box_p[..., 3:4], box_gt[..., 3:4])
        intersection = (y_2 - y_1) * (x_2 - x_1)
        xc_1 = self.min(box_p[..., 0:1], box_gt[..., 0:1])
        xc_2 = self.max(box_p[..., 2:3], box_gt[..., 2:3])
        yc_1 = self.min(box_p[..., 1:2], box_gt[..., 1:2])
        yc_2 = self.max(box_p[..., 3:4], box_gt[..., 3:4])
        c_area = (xc_2 - xc_1) * (yc_2 - yc_1)
        union = box_p_area + box_gt_area - intersection
        union = union + self.eps
        c_area = c_area + self.eps
        iou = self.div(self.cast(intersection, ms.float32), self.cast(union, ms.float32))
        res_mid0 = c_area - union
        res_mid1 = self.div(self.cast(res_mid0, ms.float32), self.cast(c_area, ms.float32))
        giou = iou - res_mid1
        giou = C.clip_by_value(giou, -1.0, 1.0)
        return giou

def xywh2x1y1x2y2(box_xywh):
    boxes_x1 = box_xywh[..., 0:1] - box_xywh[..., 2:3] / 2
    boxes_y1 = box_xywh[..., 1:2] - box_xywh[..., 3:4] / 2
    boxes_x2 = box_xywh[..., 0:1] + box_xywh[..., 2:3] / 2
    boxes_y2 = box_xywh[..., 1:2] + box_xywh[..., 3:4] / 2
    boxes_x1y1x2y2 = P.Concat(-1)((boxes_x1, boxes_y1, boxes_x2, boxes_y2))

    return boxes_x1y1x2y2

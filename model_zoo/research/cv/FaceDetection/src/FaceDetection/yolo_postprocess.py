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
"""Face detection yolov3 post-process."""
import numpy as np

from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore.common import dtype as mstype


class PtLinspace(Cell):
    '''PtLinspace'''
    def __init__(self):
        super(PtLinspace, self).__init__()
        self.tuple_to_array = P.TupleToArray()

    def construct(self, start, end, steps):
        lin_x = ()
        step = (end - start + 1) / steps
        for i in range(start, end + 1, step):
            lin_x += (i,)
        lin_x = self.tuple_to_array(lin_x)
        return lin_x


class YoloPostProcess(Cell):
    """
    Yolov3 post-process of network output.
    """
    def __init__(self, num_classes, cur_anchors, conf_thresh, network_size, reduction, anchors_mask):
        super(YoloPostProcess, self).__init__()
        self.print = P.Print()
        self.num_classes = num_classes
        self.anchors = cur_anchors
        self.conf_thresh = conf_thresh
        self.network_size = network_size
        self.reduction = reduction
        self.anchors_mask = anchors_mask
        self.num_anchors = len(anchors_mask)

        anchors_w = []
        anchors_h = []
        for i in range(len(self.anchors_mask)):
            anchors_w.append(self.anchors[i][0])
            anchors_h.append(self.anchors[i][1])
        self.anchors_w = Tensor(np.array(anchors_w).reshape((1, len(self.anchors_mask), 1)))
        self.anchors_h = Tensor(np.array(anchors_h).reshape((1, len(self.anchors_mask), 1)))

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.sigmoid = P.Sigmoid()
        self.cast = P.Cast()
        self.exp = P.Exp()
        self.concat3 = P.Concat(3)
        self.tile = P.Tile()
        self.expand_dims = P.ExpandDims()
        self.pt_linspace = PtLinspace()

    def construct(self, output):
        '''construct'''
        output_d = self.shape(output)
        num_batch = output_d[0]
        num_anchors = self.num_anchors

        num_channels = output_d[1] / num_anchors
        height = output_d[2]
        width = output_d[3]

        lin_x = self.pt_linspace(0, width - 1, width)
        lin_x = self.tile(lin_x, (height,))
        lin_x = self.cast(lin_x, mstype.float32)

        lin_y = self.pt_linspace(0, height - 1, height)
        lin_y = self.reshape(lin_y, (height, 1))
        lin_y = self.tile(lin_y, (1, width))
        lin_y = self.reshape(lin_y, (self.shape(lin_y)[0] * self.shape(lin_y)[1],))
        lin_y = self.cast(lin_y, mstype.float32)

        anchor_w = self.anchors_w
        anchor_h = self.anchors_h
        anchor_w = self.cast(anchor_w, mstype.float32)
        anchor_h = self.cast(anchor_h, mstype.float32)

        output = self.reshape(output, (num_batch, num_anchors, num_channels, height * width))

        coord_x = (self.sigmoid(output[:, :, 0, :]) + lin_x) / width
        coord_y = (self.sigmoid(output[:, :, 1, :]) + lin_y) / height
        coord_w = self.exp(output[:, :, 2, :]) * anchor_w / width
        coord_h = self.exp(output[:, :, 3, :]) * anchor_h / height
        obj_conf = self.sigmoid(output[:, :, 4, :])

        cls_conf = 0.0

        if self.num_classes > 1:
            # num_classes > 1: not implemented!
            pass

        else:
            cls_conf = self.sigmoid(output[:, :, 4, :])

        cls_scores = obj_conf * cls_conf

        coord_x_t = self.expand_dims(coord_x, 3)
        coord_y_t = self.expand_dims(coord_y, 3)
        coord_w_t = self.expand_dims(coord_w, 3)
        coord_h_t = self.expand_dims(coord_h, 3)

        coord_1 = self.concat3((coord_x_t, coord_y_t))
        coord_2 = self.concat3((coord_w_t, coord_h_t))
        coords = self.concat3((coord_1, coord_2))

        return coords, cls_scores

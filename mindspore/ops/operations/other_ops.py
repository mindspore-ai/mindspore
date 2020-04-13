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

"""Other operators."""
from ..._c_expression import signature_rw as sig_rw
from ..._c_expression import signature_kind as sig_kind
from ..._checkparam import ParamValidator as validator, Rel
from ...common import dtype as mstype
from ..primitive import Primitive, PrimitiveWithInfer, prim_attr_register


class Assign(PrimitiveWithInfer):
    """
    Assign `Parameter` with a value.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.
        - **value** (Tensor) - The value to assign.

    Outputs:
        Tensor, has the same type as original `variable`.

    Examples:
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.y = mindspore.Parameter(Tensor([1.0], mindspore.float32), name="y")
        >>>
        >>>     def construct(self, x):
        >>>         Assign()(self.y, x)
        >>>         return x
        >>> x = Tensor([2.0], mindspore.float32)
        >>> net = Net()
        >>> net(x)
    """
    __mindspore_signature__ = (
        ('variable', sig_rw.RW_WRITE, sig_kind.KIND_POSITIONAL_KEYWORD),
        ('value', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD)
    )
    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, variable, value):
        return variable

    def infer_dtype(self, variable, value):
        return variable


class BoundingBoxEncode(PrimitiveWithInfer):
    """
    Encode bounding boxes locations.

    Args:
        means (tuple): Means for encoding bounding boxes calculation. Default: (0.0, 0.0, 0.0, 0.0).
        stds (tuple): Stds for encoding bounding boxes calculation. Default: (1.0, 1.0, 1.0, 1.0).

    Inputs:
        - **anchor_box** (Tensor) - Anchor boxes.
        - **groundtruth_box** (Tensor) - Ground truth boxes.

    Outputs:
        Tensor, encoded bounding boxes.

    Examples:
        >>> boundingbox_encode = P.BoundingBoxEncode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0))
        >>> delta_box = boundingbox_encode(anchor_box, groundtruth_box)
    """

    @prim_attr_register
    def __init__(self, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
        validator.check_type('means', means, [tuple])
        validator.check_type('stds', stds, [tuple])
        validator.check("means len", len(means), '', 4)
        validator.check("stds len", len(stds), '', 4)

    def infer_shape(self, anchor_box, groundtruth_box):
        validator.check('anchor_box shape[0]', anchor_box[0], 'groundtruth_box shape[0]', groundtruth_box[0])
        validator.check('anchor_box shape[1]', anchor_box[1], '', 4)
        validator.check('groundtruth_box shape[1]', groundtruth_box[1], '', 4)
        return anchor_box

    def infer_dtype(self, anchor_box, groundtruth_box):
        args = {"anchor_box": anchor_box,
                "groundtruth_box": groundtruth_box
                }
        validator.check_type_same(args, mstype.number_type)
        return anchor_box


class BoundingBoxDecode(PrimitiveWithInfer):
    """
    Decode bounding boxes locations.

    Args:
        means (tuple): The means of deltas calculation. Default: (0.0, 0.0, 0.0, 0.0).
        stds (tuple): The standard deviations of deltas calculation. Default: (1.0, 1.0, 1.0, 1.0).
        max_shape (tuple): The max size limit for decoding box calculation.
        wh_ratio_clip (float): The limit of width and height ratio for decoding box calculation. Default: 0.016.

    Inputs:
        - **anchor_box** (Tensor) - Anchor boxes.
        - **deltas** (Tensor) - Delta of boxes.

    Outputs:
        Tensor, decoded boxes.

    Examples:
        >>> boundingbox_decode = P.BoundingBoxDecode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0),
                                                   max_shape=(768, 1280), wh_ratio_clip=0.016)
        >>> bbox = boundingbox_decode(anchor_box, deltas)
    """

    @prim_attr_register
    def __init__(self, max_shape, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0), wh_ratio_clip=0.016):
        validator.check_type('means', means, [tuple])
        validator.check_type('stds', stds, [tuple])
        validator.check_type('wh_ratio_clip', wh_ratio_clip, [float])
        validator.check("means", len(means), '', 4)
        validator.check("stds", len(stds), '', 4)
        if max_shape is not None:
            validator.check_type('max_shape', max_shape, [tuple])
            validator.check("max_shape", len(max_shape), '', 2)

    def infer_shape(self, anchor_box, deltas):
        validator.check('anchor_box shape[0]', anchor_box[0], 'deltas shape[0]', deltas[0])
        validator.check('anchor_box shape[1]', anchor_box[1], '', 4)
        validator.check('deltas shape[1]', deltas[1], '', 4)
        return anchor_box

    def infer_dtype(self, anchor_box, deltas):
        args = {"anchor_box": anchor_box,
                "deltas": deltas
                }
        validator.check_type_same(args, mstype.number_type)
        return anchor_box


class CheckValid(PrimitiveWithInfer):
    """
    Check bounding box.

    Check whether the bounding box cross data and data border.

    Inputs:
        - **bboxes** (Tensor) - Bounding boxes tensor with shape (N, 4).
        - **img_metas** (Tensor) - Raw image size information, format (height, width, ratio).

    Outputs:
        Tensor, the valided tensor.
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['bboxes', 'img_metas'], outputs=['output'])

    def infer_shape(self, bboxes_shape, metas_shape):
        validator.check_shape_length("bboxes shape length", len(bboxes_shape), 2, Rel.EQ)
        validator.check("bboxes_shape[-1]", bboxes_shape[-1], "", 4, Rel.EQ)
        validator.check_shape_length("img_metas shape length", len(metas_shape), 1, Rel.EQ)
        validator.check("img_metas shape[0]", metas_shape[0], "", 3, Rel.EQ)
        return bboxes_shape[:-1]

    def infer_dtype(self, bboxes_type, metas_type):
        return mstype.bool_


class IOU(PrimitiveWithInfer):
    r"""
    Calculate intersection over union for boxes.

    Compute the intersection over union (IOU) or the intersection over foreground (IOF) based on the ground-truth and
    predicted regions.

    .. math::
        \text{IOU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}

        \text{IOF} = \frac{\text{Area of Overlap}}{\text{Area of Ground Truth}}

    Args:
        mode (string): The mode is used to specify the calculation method,
                       now support 'iou' (intersection over union) or 'iof'
                       (intersection over foreground) mode. Default: 'iou'.

    Inputs:
        - **anchor_boxes** (Tensor) - Anchor boxes, tensor of shape (N, 4). "N" indicates the number of anchor boxes,
          and the value "4" refers to "x0", "x1", "y0", and "y1".
        - **gt_boxes** (Tensor) - Ground truth boxes, tensor of shape (M, 4). "M" indicates the number of ground
          truth boxes, and the value "4" refers to "x0", "x1", "y0", and "y1".

    Outputs:
        Tensor, the 'iou' values, tensor of shape (M, N).

    Raises:
        KeyError: When `mode` is not 'iou' or 'iof'.

    Examples:
        >>> iou = P.IOU()
        >>> anchor_boxes = Tensor(np.random.randint(1,5, [10, 4]))
        >>> gt_boxes = Tensor(np.random.randint(1,5, [3, 4]))
        >>> iou(anchor_boxes, gt_boxes)
    """

    @prim_attr_register
    def __init__(self, mode='iou'):
        if mode not in {'iou', 'iof'}:
            raise KeyError("Mode only support 'iou' or 'iof'.")
        self.init_prim_io_names(inputs=['anchor_boxes', 'gt_boxes'], outputs=['overlap'])

    def infer_shape(self, anchor_boxes, gt_boxes):
        validator.check('gt_boxes shape[1]', gt_boxes[1], '', 4)
        validator.check('anchor_boxes shape[1]', anchor_boxes[1], '', 4)
        validator.check('anchor_boxes rank', len(anchor_boxes), '', 2)
        validator.check('gt_boxes rank', len(gt_boxes), '', 2)
        iou = [gt_boxes[0], anchor_boxes[0]]
        return iou

    def infer_dtype(self, anchor_boxes, gt_boxes):
        validator.check_subclass("anchor_boxes", anchor_boxes, mstype.tensor)
        validator.check_subclass("gt_boxes", gt_boxes, mstype.tensor)
        args = {"anchor_boxes": anchor_boxes, "gt_boxes": gt_boxes}
        validator.check_type_same(args, (mstype.float16,))
        return anchor_boxes


class MakeRefKey(Primitive):
    """
    Make a RefKey instance by string. RefKey stores the name of Parameter, can be passed through the functions,
    and used for Assign target.

    Args:
        tag (str): Parameter name to make the RefKey.

    Inputs:
        No input.

    Outputs:
        RefKeyType, made from the Parameter name.

    Examples:
        >>> from mindspore.ops import functional as F
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.y = mindspore.Parameter(Tensor(np.ones([6, 8, 10]), mindspore.int32), name="y")
        >>>         self.make_ref_key = P.MakeRefKey("y")
        >>>
        >>>     def construct(self, x):
        >>>         key = self.make_ref_key()
        >>>         ref = F.make_ref(key, x, self.y)
        >>>         return ref * x
        >>>
        >>> x = Tensor(np.ones([3, 4, 5]), mindspore.int32)
        >>> net = Net()
        >>> net(x)
    """

    @prim_attr_register
    def __init__(self, tag):
        validator.check_type('tag', tag, (str,))

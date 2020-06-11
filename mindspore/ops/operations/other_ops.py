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
import functools
from ..._c_expression import signature_rw as sig_rw
from ..._c_expression import signature_kind as sig_kind
from ..._c_expression import signature_dtype as sig_dtype
from ..._checkparam import Validator as validator, Rel
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
        >>>         P.Assign()(self.y, x)
        >>>         return x
        >>> x = Tensor([2.0], mindspore.float32)
        >>> net = Net()
        >>> net(x)
    """
    __mindspore_signature__ = (
        ('variable', sig_rw.RW_WRITE, sig_kind.KIND_POSITIONAL_KEYWORD, sig_kind.KIND_EMPTY_DEFAULT_VALUE, sig_dtype.T),
        ('value', sig_rw.RW_READ, sig_kind.KIND_POSITIONAL_KEYWORD, sig_kind.KIND_EMPTY_DEFAULT_VALUE, sig_dtype.T)
    )
    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, variable, value):
        return variable

    def infer_dtype(self, variable, value):
        args = {"variable": variable, "value": value}
        validator.check_tensor_type_same(args, (mstype.bool_,) + mstype.number_type, self.name)
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
        >>> anchor_box = Tensor([[4,1,2,1],[2,2,2,3]],mindspore.float32)
        >>> groundtruth_box = Tensor([[3,1,2,2],[1,2,1,4]],mindspore.float32)
        >>> boundingbox_encode = P.BoundingBoxEncode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0))
        >>> boundingbox_encode(anchor_box, groundtruth_box)
        [[5.0000000e-01  5.0000000e-01  -6.5504000e+04  6.9335938e-01]
         [-1.0000000e+00  2.5000000e-01  0.0000000e+00  4.0551758e-01]]

    """

    @prim_attr_register
    def __init__(self, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
        validator.check_value_type('means', means, [tuple], self.name)
        validator.check_value_type('stds', stds, [tuple], self.name)
        validator.check_integer("means len", len(means), 4, Rel.EQ, self.name)
        validator.check_integer("stds len", len(stds), 4, Rel.EQ, self.name)

    def infer_shape(self, anchor_box, groundtruth_box):
        validator.check('anchor_box shape[0]', anchor_box[0], 'groundtruth_box shape[0]', groundtruth_box[0], Rel.EQ,
                        self.name)
        validator.check_integer('anchor_box shape[1]', anchor_box[1], 4, Rel.EQ, self.name)
        validator.check_integer('groundtruth_box shape[1]', groundtruth_box[1], 4, Rel.EQ, self.name)
        return anchor_box

    def infer_dtype(self, anchor_box, groundtruth_box):
        args = {"anchor_box": anchor_box, "groundtruth_box": groundtruth_box}
        validator.check_tensor_type_same(args, mstype.number_type, self.name)
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
        >>> anchor_box = Tensor([[4,1,2,1],[2,2,2,3]],mindspore.float32)
        >>> deltas = Tensor([[3,1,2,2],[1,2,1,4]],mindspore.float32)
        >>> boundingbox_decode = P.BoundingBoxDecode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0),
        >>>                                          max_shape=(768, 1280), wh_ratio_clip=0.016)
        >>> boundingbox_decode(anchor_box, deltas)
        [[4.1953125  0.  0.  5.1953125]
         [2.140625  0.  3.859375  60.59375]]

    """

    @prim_attr_register
    def __init__(self, max_shape, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0), wh_ratio_clip=0.016):
        validator.check_value_type('means', means, [tuple], self.name)
        validator.check_value_type('stds', stds, [tuple], self.name)
        validator.check_value_type('wh_ratio_clip', wh_ratio_clip, [float], self.name)
        validator.check_integer("means len", len(means), 4, Rel.EQ, self.name)
        validator.check_integer("stds len", len(stds), 4, Rel.EQ, self.name)
        if max_shape is not None:
            validator.check_value_type('max_shape', max_shape, [tuple], self.name)
            validator.check_integer("max_shape len", len(max_shape), 2, Rel.EQ, self.name)

    def infer_shape(self, anchor_box, deltas):
        validator.check('anchor_box shape[0]', anchor_box[0], 'deltas shape[0]', deltas[0], Rel.EQ, self.name)
        validator.check_integer('anchor_box shape[1]', anchor_box[1], 4, Rel.EQ, self.name)
        validator.check_integer('deltas shape[1]', deltas[1], 4, Rel.EQ, self.name)
        return anchor_box

    def infer_dtype(self, anchor_box, deltas):
        args = {"anchor_box": anchor_box, "deltas": deltas}
        validator.check_tensor_type_same(args, mstype.number_type, self.name)
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

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.ops import operations as P
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.check_valid = P.CheckValid()
        >>>     def construct(self, x, y):
        >>>         valid_result = self.check_valid(x, y)
        >>>         return valid_result
        >>>
        >>> bboxes = Tensor(np.linspace(0, 6, 12).reshape(3, 4), mindspore.float32)
        >>> img_metas = Tensor(np.array([2, 1, 3]), mindspore.float32)
        >>> net = Net()
        >>> result = net(bboxes, img_metas)
        [True   False   False]
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['bboxes', 'img_metas'], outputs=['output'])

    def infer_shape(self, bboxes_shape, metas_shape):
        validator.check("bboxes rank", len(bboxes_shape), "", 2, Rel.EQ, self.name)
        validator.check("bboxes_shape[-1]", bboxes_shape[-1], "", 4, Rel.EQ, self.name)
        validator.check("img_metas rank", len(metas_shape), "", 1, Rel.EQ, self.name)
        validator.check("img_metas shape[0]", metas_shape[0], "", 3, Rel.EQ, self.name)
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
        >>> anchor_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float32)
        >>> gt_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float32)
        >>> iou(anchor_boxes, gt_boxes)
    """

    @prim_attr_register
    def __init__(self, mode='iou'):
        if mode not in {'iou', 'iof'}:
            raise KeyError("Mode only support 'iou' or 'iof'.")
        self.init_prim_io_names(inputs=['anchor_boxes', 'gt_boxes'], outputs=['overlap'])

    def infer_shape(self, anchor_boxes, gt_boxes):
        validator.check_integer('gt_boxes shape[1]', gt_boxes[1], 4, Rel.EQ, self.name)
        validator.check_integer('anchor_boxes shape[1]', anchor_boxes[1], 4, Rel.EQ, self.name)
        validator.check_integer('anchor_boxes rank', len(anchor_boxes), 2, Rel.EQ, self.name)
        validator.check_integer('gt_boxes rank', len(gt_boxes), 2, Rel.EQ, self.name)
        iou = [gt_boxes[0], anchor_boxes[0]]
        return iou

    def infer_dtype(self, anchor_boxes, gt_boxes):
        args = {"anchor_boxes": anchor_boxes, "gt_boxes": gt_boxes}
        validator.check_tensor_type_same(args, (mstype.float16,), self.name)
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
        validator.check_value_type('tag', tag, (str,), self.name)

    def __call__(self):
        pass


class Partial(Primitive):
    """
    Make a partial function instance, used for pynative mode.

    Inputs:
        - **args** (Union[FunctionType, Tensor]) - The function and bind arguments.

    Outputs:
        FunctionType, partial function binded with arguments.
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __call__(self, *args):
        func = args[0].__call__
        partial_func = functools.partial(func, *args[1:])
        return partial_func

class Depend(Primitive):
    """
    Depend is used for process side-effect operations.

    Inputs:
        - **value** (Tensor) - the real value to return for depend operator.
        - **expr** (Expression) - the expression to execute with no outputs.

    Outputs:
        Tensor, the value passed by last operator.
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __call__(self, value, expr):
        return value


class CheckBprop(PrimitiveWithInfer):
    """
    Checks whether data type and shape of corresponding element from tuple x and y are the same.

    Raises:
        TypeError: If not the same.

    Inputs:
        - **input_x** (tuple[Tensor]) - The input_x contains the outputs of bprop to be checked.
        - **input_y** (tuple[Tensor]) - The input_y contains the inputs of bprop to check against.

    Outputs:
        (tuple[Tensor]), the input_x,
        if data type and shape of corresponding elements from `input_x` and `input_y` are the same.

    Examples:
        >>> input_x = (Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32),)
        >>> input_y = (Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32),)
        >>> out = P.CheckBprop()(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self, prim_to_check=""):
        """init CheckBprop"""
        self.prim_to_check = prim_to_check

    def infer_shape(self, xshapes, yshapes):
        tips = f'Bprop of {self.prim_to_check}'
        if len(xshapes) < len(yshapes):
            raise TypeError(f"{tips}, the size of output should be {len(yshapes)},"
                            f" but got {len(xshapes)}.")
        checking_range = len(yshapes)
        for i in range(checking_range):
            xshape = xshapes[i]
            yshape = yshapes[i]
            if not xshape or not yshape:
                continue
            if xshape != yshape:
                raise TypeError(f"{tips}, the shape of {i}th output should be {yshape},"
                                f" but got {xshape}.")
        return xshapes

    def infer_dtype(self, xdtypes, ydtypes):
        tips = f'Bprop of {self.prim_to_check}'
        if len(xdtypes) < len(ydtypes):
            raise TypeError(f"{tips}, the size of output should be {len(ydtypes)},"
                            f" but got {len(xdtypes)}.")
        checking_range = len(ydtypes)
        for i in range(checking_range):
            xdtype = xdtypes[i]
            ydtype = ydtypes[i]
            if isinstance(xdtype, mstype.anything_type) or isinstance(ydtype, mstype.anything_type):
                continue
            if isinstance(ydtype, mstype.function_type):
                if not isinstance(xdtype, mstype.env_type_type):
                    raise TypeError(f"{tips}, the dtype of {i}th output should be {mstype.env_type_type},"
                                    f" but got {xdtype}.")
                continue
            if xdtype != ydtype:
                raise TypeError(f"{tips}, the dtype of {i}th output should be {ydtype},"
                                f" but got {xdtype}.")
        return xdtypes


class ConfusionMatrix(PrimitiveWithInfer):
    r"""
    Calculate the confusion matrix from labels and predictions.

    Args:
        num_classes (int): The num of classes.
        dtype (str): Data type of confusion matrix. Default: 'int32'.

    Inputs:
        - **labels** (Tensor) - real labels, tensor of 1-D. the dtype must be non-negative Integer.
        - **predictions** (Tensor) - the labels from prediction, tensor of 1-D.
          the shape same as `labels` and the dtype must be non-negative Integer.
        - **weights** (Tensor) - tensor of 1-D. the shape same as `predictions`.

    Outputs:
        Tensor, the confusion matrix, with shape (`num_classes`, `num_classes`).

    Examples:
        >>> confusion_matrix = P.ConfusionMatrix(4)
        >>> labels = Tensor([0, 1, 1, 3], mindspore.int32)
        >>> predictions = Tensor([1, 2, 1, 3], mindspore.int32)
        >>> confusion_matrix(labels, predictions)
    """

    @prim_attr_register
    def __init__(self, num_classes, dtype="int32"):
        validator.check_value_type("num_classes", num_classes, [int], self.name)
        validator.check_value_type("dtype", dtype, [str], self.name)

    def infer_shape(self, labels, predictions, weights=None):
        validator.check('labels dimension', len(labels), '', 1, Rel.EQ, self.name)
        validator.check('labels shape', labels, 'predictions shape', predictions, Rel.EQ, self.name)
        if weights is not None:
            validator.check('labels shape', labels, 'weights shape', weights, Rel.EQ, self.name)
        ret = (self.num_classes, self.num_classes)
        return ret

    def infer_dtype(self, labels, predictions, weights=None):
        validator.check_subclass('labels', labels, mstype.tensor, self.name)
        validator.check_subclass('predictions', predictions, mstype.tensor, self.name)
        if weights is not None:
            validator.check_subclass('weights', weights, mstype.tensor, self.name)
        args = {"labels": labels, "predictions": predictions}
        validator.check_tensor_type_same(args, (mstype.number_type), self.name)
        return labels

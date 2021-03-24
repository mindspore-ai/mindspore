# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from mindspore.common import monad
from .. import signature as sig
from ..._checkparam import Validator as validator, Rel
from ...common import dtype as mstype
from ..primitive import Primitive, PrimitiveWithCheck, PrimitiveWithInfer, prim_attr_register


class Assign(PrimitiveWithCheck):
    """
    Assigns `Parameter` with a value.

    Inputs of `variable` and `value` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.
        - **value** (Tensor) - The value to be assigned.

    Outputs:
        Tensor, has the same type as original `variable`.

    Raises:
        TypeError: If `variable` is not a Parameter.
        TypeError: If `value` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.y = mindspore.Parameter(Tensor([1.0], mindspore.float32), name="y")
        ...
        ...     def construct(self, x):
        ...         ops.Assign()(self.y, x)
        ...         return self.y
        ...
        >>> x = Tensor([2.0], mindspore.float32)
        >>> net = Net()
        >>> output = net(x)
        >>> print(output)
        Parameter (name=y, shape=(1,), dtype=Float32, requires_grad=True)
    """
    __mindspore_signature__ = (
        sig.make_sig('variable', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('value', dtype=sig.sig_dtype.T),
        sig.make_sig('u', default=monad.U, dtype=sig.sig_dtype.T1)
    )

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['ref', 'value'], outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)

    def check_dtype(self, variable, value):
        types = mstype.number_type + (mstype.bool_,)
        if variable != mstype.type_refkey:
            validator.check_tensor_dtype_valid("variable", variable, types, self.name)
        validator.check_scalar_or_tensor_types_same({"value": value}, types, self.name)


class InplaceAssign(PrimitiveWithInfer):
    """
    Inplace assign `Parameter` with a value.
    This primitive can only use in graph kernel.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.
        - **value** (Tensor) - The value to be assigned.
        - **depend** (Tensor) - The dependent tensor to keep this op connected in graph.

    Outputs:
        Tensor, has the same type as original `variable`.

    Raises:
        TypeError: If `value` or `depend` is not a Tensor.

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.inplace_assign = ops.InplaceAssign()
        ...
        ...     def construct(self, x):
        ...         val = x - 1.0
        ...         ret = x + 2.0
        ...         return self.inplace_assign(x, val, ret)
        ...
        >>> x = Tensor([2.0], mindspore.float32)
        >>> net = Net()
        >>> output = net(x)
        >>> print(output)
   """
    @ prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'y', 'z'], outputs=['output'])

    def infer_shape(self, x, y, z):
        return z

    def infer_dtype(self, x, y, z):
        return z

class Load(PrimitiveWithCheck):
    """
    Load `Parameter` to a value.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.

    Outputs:
        Tensor - The loaded parameter tensor value.
    """
    __mindspore_signature__ = (
        sig.make_sig('variable', sig.sig_rw.RW_READ, dtype=sig.sig_dtype.T),
        sig.make_sig('u', dtype=sig.sig_dtype.T1)
    )

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['ref', 'u'], outputs=['output'])

    def check_dtype(self, variable):
        if variable != mstype.type_refkey:
            validator.check_tensor_type_same({"variable": variable}, mstype.number_type, self.name)

class BoundingBoxEncode(PrimitiveWithInfer):
    """
    Encodes bounding boxes locations.

    Args:
        means (tuple): Means for encoding bounding boxes calculation. Default: (0.0, 0.0, 0.0, 0.0).
        stds (tuple): The standard deviations of deltas calculation. Default: (1.0, 1.0, 1.0, 1.0).

    Inputs:
        - **anchor_box** (Tensor) - Anchor boxes. The shape of anchor_box must be (n, 4).
        - **groundtruth_box** (Tensor) - Ground truth boxes. Which has the same shape with anchor_box.

    Outputs:
        Tensor, encoded bounding boxes.

    Raises:
        TypeError: If `means` or `stds` is not a tuple.
        TypeError: If `anchor_box` or `groundtruth_box` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> anchor_box = Tensor([[2, 2, 2, 3], [2, 2, 2, 3]], mindspore.float32)
        >>> groundtruth_box = Tensor([[1, 2, 1, 4], [1, 2, 1, 4]], mindspore.float32)
        >>> boundingbox_encode = ops.BoundingBoxEncode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0))
        >>> output = boundingbox_encode(anchor_box, groundtruth_box)
        >>> print(output)
        [[ -1.  0.25  0.  0.40551758]
         [ -1.  0.25  0.  0.40551758]]
    """

    @prim_attr_register
    def __init__(self, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
        validator.check_value_type('means', means, (tuple), self.name)
        validator.check_value_type('stds', stds, (tuple), self.name)
        for i, value in enumerate(means):
            validator.check_value_type("means[%d]" % i, value, [float], self.name)
        for i, value in enumerate(stds):
            validator.check_value_type("stds[%d]" % i, value, [float], self.name)
        validator.check_equal_int(len(means), 4, "means len", self.name)
        validator.check_equal_int(len(stds), 4, "stds len", self.name)

    def infer_shape(self, anchor_box, groundtruth_box):
        validator.check('anchor_box shape[0]', anchor_box[0], 'groundtruth_box shape[0]', groundtruth_box[0], Rel.EQ,
                        self.name)
        validator.check("anchor_box rank", len(anchor_box), "", 2, Rel.EQ, self.name)
        validator.check("groundtruth_box rank", len(groundtruth_box), "", 2, Rel.EQ, self.name)
        validator.check_equal_int(anchor_box[1], 4, 'anchor_box shape[1]', self.name)
        validator.check_equal_int(groundtruth_box[1], 4, 'groundtruth_box shape[1]', self.name)
        return anchor_box

    def infer_dtype(self, anchor_box, groundtruth_box):
        args = {"anchor_box": anchor_box, "groundtruth_box": groundtruth_box}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return anchor_box


class BoundingBoxDecode(PrimitiveWithInfer):
    """
    Decodes bounding boxes locations.

    Args:
        means (tuple): The means of deltas calculation. Default: (0.0, 0.0, 0.0, 0.0).
        stds (tuple): The standard deviations of deltas calculation. Default: (1.0, 1.0, 1.0, 1.0).
        max_shape (tuple): The max size limit for decoding box calculation.
        wh_ratio_clip (float): The limit of width and height ratio for decoding box calculation. Default: 0.016.

    Inputs:
        - **anchor_box** (Tensor) - Anchor boxes. The shape of `anchor_box` must be (n, 4).
        - **deltas** (Tensor) - Delta of boxes. Which has the same shape with `anchor_box`.

    Outputs:
        Tensor, decoded boxes.

    Raises:
        TypeError: If `means`, `stds` or `max_shape` is not a tuple.
        TypeError: If `wh_ratio_clip` is not a float.
        TypeError: If `anchor_box` or `deltas` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> anchor_box = Tensor([[4, 1, 2, 1], [2, 2, 2, 3]], mindspore.float32)
        >>> deltas = Tensor([[3, 1, 2, 2], [1, 2, 1, 4]], mindspore.float32)
        >>> boundingbox_decode = ops.BoundingBoxDecode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0),
        ...                                          max_shape=(768, 1280), wh_ratio_clip=0.016)
        >>> output = boundingbox_decode(anchor_box, deltas)
        >>> print(output)
        [[ 4.1953125  0.         0.         5.1953125]
         [ 2.140625   0.         3.859375  60.59375  ]]

    """

    @prim_attr_register
    def __init__(self, max_shape, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0), wh_ratio_clip=0.016):
        validator.check_value_type('means', means, (tuple), self.name)
        validator.check_value_type('stds', stds, (tuple), self.name)
        for i, value in enumerate(means):
            validator.check_value_type("means[%d]" % i, value, [float], self.name)
        for i, value in enumerate(stds):
            validator.check_value_type("stds[%d]" % i, value, [float], self.name)
        validator.check_value_type('wh_ratio_clip', wh_ratio_clip, [float], self.name)
        validator.check_equal_int(len(means), 4, "means len", self.name)
        validator.check_equal_int(len(stds), 4, "stds len", self.name)
        if max_shape is not None:
            validator.check_value_type('max_shape', max_shape, [tuple], self.name)
            validator.check_equal_int(len(max_shape), 2, "max_shape len", self.name)

    def infer_shape(self, anchor_box, deltas):
        validator.check('anchor_box shape[0]', anchor_box[0], 'deltas shape[0]', deltas[0], Rel.EQ, self.name)
        validator.check("anchor_box rank", len(anchor_box), "", 2, Rel.EQ, self.name)
        validator.check("deltas rank", len(deltas), "", 2, Rel.EQ, self.name)
        validator.check_equal_int(anchor_box[1], 4, 'anchor_box shape[1]', self.name)
        validator.check_equal_int(deltas[1], 4, 'deltas shape[1]', self.name)
        return anchor_box

    def infer_dtype(self, anchor_box, deltas):
        args = {"anchor_box": anchor_box, "deltas": deltas}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return anchor_box


class CheckValid(PrimitiveWithInfer):
    """
    Checks bounding box.

    Checks whether the bounding box cross data and data border are valid.

    Inputs:
        - **bboxes** (Tensor) - Bounding boxes tensor with shape (N, 4). Data type must be float16 or float32.
        - **img_metas** (Tensor) - Raw image size information with the format of (height, width, ratio).
          Data type must be float16 or float32.

    Outputs:
        Tensor, with shape of (N,) and dtype of bool.

    Raises:
        TypeError: If `bboxes` or `img_metas` is not a Tensor.
        TypeError: If dtype of `bboxes` or `img_metas` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.ops import operations as ops
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.check_valid = ops.CheckValid()
        ...     def construct(self, x, y):
        ...         valid_result = self.check_valid(x, y)
        ...         return valid_result
        ...
        >>> bboxes = Tensor(np.linspace(0, 6, 12).reshape(3, 4), mindspore.float32)
        >>> img_metas = Tensor(np.array([2, 1, 3]), mindspore.float32)
        >>> net = Net()
        >>> output = net(bboxes, img_metas)
        >>> print(output)
        [ True False False]
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
        valid_type = [mstype.float32, mstype.float16, mstype.int16, mstype.uint8]
        validator.check_tensor_dtype_valid("bboxes_type", bboxes_type, valid_type, self.name)
        validator.check_tensor_dtype_valid("metas_type", metas_type, valid_type, self.name)
        return mstype.bool_


class IOU(PrimitiveWithInfer):
    r"""
    Calculates intersection over union for boxes.

    Computes the intersection over union (IOU) or the intersection over foreground (IOF) based on the ground-truth and
    predicted regions.

    .. math::
        \text{IOU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}

        \text{IOF} = \frac{\text{Area of Overlap}}{\text{Area of Ground Truth}}

    Args:
        mode (string): The mode is used to specify the calculation method,
                       now supporting 'iou' (intersection over union) or 'iof'
                       (intersection over foreground) mode. Default: 'iou'.

    Inputs:
        - **anchor_boxes** (Tensor) - Anchor boxes, tensor of shape (N, 4). "N" indicates the number of anchor boxes,
          and the value "4" refers to "x0", "y0", "x1", and "y1". Data type must be float16 or float32.
        - **gt_boxes** (Tensor) - Ground truth boxes, tensor of shape (M, 4). "M" indicates the number of ground
          truth boxes, and the value "4" refers to "x0", "y0", "x1", and "y1". Data type must be float16 or float32.

    Outputs:
        Tensor, the 'iou' values, tensor of shape (M, N), with the same data type as `anchor_boxes`.

    Raises:
        KeyError: When `mode` is not 'iou' or 'iof'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> iou = ops.IOU()
        >>> anchor_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
        >>> gt_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
        >>> output = iou(anchor_boxes, gt_boxes)
        >>> print(output.shape)
        (3, 3)
    """

    @prim_attr_register
    def __init__(self, mode='iou'):
        if mode not in {'iou', 'iof'}:
            raise KeyError("Mode only support 'iou' or 'iof'.")
        self.init_prim_io_names(inputs=['anchor_boxes', 'gt_boxes'], outputs=['overlap'])

    def infer_shape(self, anchor_boxes, gt_boxes):
        validator.check_equal_int(gt_boxes[1], 4, 'gt_boxes shape[1]', self.name)
        validator.check_equal_int(anchor_boxes[1], 4, 'anchor_boxes shape[1]', self.name)
        validator.check_equal_int(len(anchor_boxes), 2, 'anchor_boxes rank', self.name)
        validator.check_equal_int(len(gt_boxes), 2, 'gt_boxes rank', self.name)
        iou = [gt_boxes[0], anchor_boxes[0]]
        return iou

    def infer_dtype(self, anchor_boxes, gt_boxes):
        valid_type = [mstype.float32, mstype.float16]
        validator.check_tensor_dtype_valid("anchor_boxes", anchor_boxes, valid_type, self.name)
        validator.check_tensor_dtype_valid("gt_boxes", gt_boxes, valid_type, self.name)
        return anchor_boxes


class Partial(Primitive):
    """
    Makes a partial function instance, used for pynative mode.

    Inputs:
        - **args** (Union[FunctionType, Tensor]) - The function and bind arguments.

    Outputs:
        FunctionType, partial function binded with arguments.
    """

    # Side effect will propagated from the first argument to return value.
    side_effect_propagate = 1

    @prim_attr_register
    def __init__(self):
        self.add_prim_attr('side_effect_propagate', 1)

    def __call__(self, *args):
        func = args[0].__call__
        partial_func = functools.partial(func, *args[1:])
        return partial_func


class Depend(Primitive):
    """
    Depend is used for processing dependency operations.

    In most scenarios, if operators have IO side effects or memory side effects,
    they will be executed according to the user's semantics. In some scenarios,
    if the two operators A and B have no order dependency, and A must be executed
    before B, we recommend using Depend to specify their execution order. The
    usage method is as follows::

        a = A(x)                --->        a = A(x)
        b = B(y)                --->        y = Depend(y, a)
                                --->        b = B(y)

    Inputs:
        - **value** (Tensor) - the real value to return for depend operator.
        - **expr** (Expression) - the expression to execute with no outputs.

    Outputs:
        Tensor, the value passed by last operator.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.ops.operations as P
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.softmax = P.Softmax()
        ...         self.depend = P.Depend()
        ...
        ...     def construct(self, x, y):
        ...         mul = x * y
        ...         y = self.depend(y, mul)
        ...         ret = self.softmax(y)
        ...         return ret
        ...
        >>> x = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
        >>> y = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
        >>> net = Net()
        >>> output = net(x, y)
        >>> print(output)
        [[0.2 0.2 0.2 0.2 0.2]
         [0.2 0.2 0.2 0.2 0.2]
         [0.2 0.2 0.2 0.2 0.2]
         [0.2 0.2 0.2 0.2 0.2]]
    """

    # Side effect will propagated from the first argument to return value.
    side_effect_propagate = 1

    @prim_attr_register
    def __init__(self):
        self.add_prim_attr('side_effect_propagate', 1)

    def __call__(self, value, expr):
        return value

class UpdateState(Primitive):
    """
    UpdateState is used for update side-effect state.

    Inputs:
        - **value** (State) - the state value to be updated.
        - **expr** (Expression) - the expression to evaluate before state changes.

    Outputs:
        State, the updated state value.
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __call__(self, state, expr):
        return state

class CheckBprop(PrimitiveWithInfer):
    """
    Checks whether the data type and the shape of corresponding elements from tuples x and y are the same.

    Inputs:
        - **input_x** (tuple[Tensor]) - The `input_x` contains the outputs of bprop to be checked.
        - **input_y** (tuple[Tensor]) - The `input_y` contains the inputs of bprop to check against.

    Outputs:
        (tuple[Tensor]), the `input_x`,
        if data type and shape of corresponding elements from `input_x` and `input_y` are the same.

    Raises:
        TypeError: If `input_x` or `input_y` is not a Tensor.

    Examples:
        >>> input_x = (Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32),)
        >>> input_y = (Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32),)
        >>> out = ops.CheckBprop()(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self, prim_to_check=""):
        """Initialize CheckBprop"""
        self.prim_to_check = prim_to_check

    def infer_shape(self, xshapes, yshapes):
        tips = f'Bprop of {self.prim_to_check}'
        validator.check_value_type('grads', xshapes, (tuple,), tips)
        validator.check_value_type('params', yshapes, (tuple,), tips)
        if len(xshapes) < len(yshapes):
            raise ValueError(f"{tips}, the size of output should be {len(yshapes)},"
                             f" but got {len(xshapes)}.")
        checking_range = len(yshapes)
        for i in range(checking_range):
            xshape = xshapes[i]
            yshape = yshapes[i]
            if not xshape or not yshape:
                continue
            if xshape != yshape:
                raise ValueError(f"{tips}, the shape of {i}th output should be {yshape},"
                                 f" but got {xshape}.")
        return xshapes

    def infer_dtype(self, xdtypes, ydtypes):
        tips = f'Bprop of {self.prim_to_check}'
        validator.check_value_type('grads', xdtypes, (tuple,), tips)
        validator.check_value_type('params', ydtypes, (tuple,), tips)
        if len(xdtypes) < len(ydtypes):
            raise ValueError(f"{tips}, the size of output should be {len(ydtypes)},"
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
    Calculates the confusion matrix from labels and predictions.

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

    Raises:
        TypeError: If `num_classes` is not an int.
        TypeError: If `dtype` is not a str.
        TypeError: If `labels`, `predictions` or weight` is not a Tensor.

    Examples:
        >>> confusion_matrix = ops.ConfusionMatrix(4)
        >>> labels = Tensor([0, 1, 1, 3], mindspore.int32)
        >>> predictions = Tensor([1, 2, 1, 3], mindspore.int32)
        >>> output = confusion_matrix(labels, predictions)
        >>> print(output)
        [[0 1 0 0]
         [0 1 1 0]
         [0 0 0 0]
         [0 0 0 1]]
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
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.number_type), self.name)
        return labels


class PopulationCount(PrimitiveWithInfer):
    r"""
    Calculates population count.

    Inputs:
        - **input** (Tensor) -  The data type must be int16 or uint16.

    Outputs:
        Tensor, with the same shape as the input.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> population_count = ops.PopulationCount()
        >>> x_input = Tensor([0, 1, 3], mindspore.int16)
        >>> output = population_count(x_input)
        >>> print(output)
        [0 1 2]
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, (mstype.int16, mstype.uint16,), self.name)
        return mstype.tensor_type(mstype.uint8)


class Push(PrimitiveWithInfer):
    """
    Pushes the inputs of the corresponding optimizer to parameter server.

    Args:
        optim_type (string): The optimizer type. Default: 'ApplyMomentum'.
        only_shape_indices (list): The indices of input of which only shape
                                   will be pushed to parameter server. Default: None.

    Inputs:
        - **optim_inputs** (tuple) - The inputs for this kind of optimizer.
        - **optim_input_shapes** (tuple) - The shapes of the inputs.

    Outputs:
        Tensor, the key of the weight which needs to be updated.
    """

    @prim_attr_register
    def __init__(self, optim_type='ApplyMomentum', only_shape_indices=None):
        """Initialize Push"""
        self.add_prim_attr("primitive_target", "CPU")
        self.add_prim_attr("_side_effect", True)
        self.init_prim_io_names(inputs=['optim_inputs', 'optim_input_shapes'], outputs=['key'])

    def infer_shape(self, inputs, shapes):
        return [1]

    def infer_dtype(self, inputs, shapes):
        return mstype.uint64


class Pull(PrimitiveWithInfer):
    """
    Pulls weight from parameter server.

    Inputs:
        - **key** (Tensor) - The key of the weight.
        - **weight** (Tensor) - The weight to be updated.

    Outputs:
        None.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Pull"""
        self.add_prim_attr("primitive_target", "CPU")
        self.init_prim_io_names(inputs=['key', 'weight'], outputs=['output'])

    def infer_shape(self, key_shape, weight_shape):
        return [1]

    def infer_dtype(self, key_dtype, weight_dtype):
        return mstype.float32


class identity(Primitive):
    """
    Makes a identify primitive, used for pynative mode.

    Inputs:
        - **x** (Any) - identity input value.

    Outputs:
        The same as input.
    """

    # Side effect will propagated from the first argument to return value.
    side_effect_propagate = 1

    @prim_attr_register
    def __init__(self):
        self.add_prim_attr('side_effect_propagate', 1)

    def __call__(self, x):
        return x

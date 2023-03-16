# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore import log as logger
from mindspore.ops import signature as sig
from mindspore._checkparam import Validator as validator, Rel
from mindspore.common import dtype as mstype
from mindspore.ops.primitive import Primitive, PrimitiveWithCheck, PrimitiveWithInfer, prim_attr_register
from mindspore.ops.operations._pyfunc_registry import add_pyfunc


class Assign(Primitive):
    """
    Assigns `Parameter` with a value.

    Refer to :func:`mindspore.ops.assign` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> value = Tensor([2.0], mindspore.float32)
        >>> variable = mindspore.Parameter(Tensor([1.0], mindspore.float32), name="variable")
        >>> assign = ops.Assign()
        >>> x = assign(variable, value)
        >>> print(variable.asnumpy())
        [2.]
    """
    __mindspore_signature__ = (
        sig.make_sig('variable', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('value', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize Assign."""
        self.init_prim_io_names(inputs=['ref', 'value'], outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)


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
        """Initialize Load."""
        self.init_prim_io_names(inputs=['ref', 'u'], outputs=['output'])

    def check_dtype(self, variable):
        if variable != mstype.type_refkey:
            validator.check_tensors_dtypes_same_and_valid({"variable": variable}, mstype.number_type, self.name)


class _DynamicLossScale(PrimitiveWithInfer):
    """
    Dynamic multi layer loss scale operator.

    Inputs:
        - **input_x** (Tensor) - Output of last operator.
        - **loss_scale** (Tensor) - Dynamic loss scale.

    Outputs:
        Tensor - The same as `input_x`.
    """
    __mindspore_signature__ = (
        sig.make_sig('input_x', dtype=sig.sig_dtype.T),
        sig.make_sig('loss_scale', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, layer=-1):
        """Initialize DynamicLossScale."""
        validator.check_value_type('layer', layer, (int,), self.name)
        self.init_prim_io_names(inputs=['input_x', 'loss_scale'], outputs=['output'])

    def infer_shape(self, input_x, loss_scale):
        return input_x

    def infer_dtype(self, input_x, loss_scale):
        return input_x


class BoundingBoxEncode(PrimitiveWithInfer):
    """
    Encodes bounding boxes locations.

    This operator will calculate the offset between the predicted bounding boxes and the real bounding boxes,
    and this offset will be used as a variable for the loss.

    Args:
        means (tuple): Means for encoding bounding boxes calculation. Default: (0.0, 0.0, 0.0, 0.0).
        stds (tuple): The standard deviations of deltas calculation. Default: (1.0, 1.0, 1.0, 1.0).

    Inputs:
        - **anchor_box** (Tensor) - Anchor boxes. The shape of anchor_box must be (n, 4).
        - **groundtruth_box** (Tensor) - Ground truth boxes. Which has the same shape with anchor_box.

    Outputs:
        Tensor, encoded bounding boxes. It has the same data type and shape as input `anchor_box`.

    Raises:
        TypeError: If `means` or `stds` is not a tuple.
        TypeError: If `anchor_box` or `groundtruth_box` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        """Initialize BoundingBoxEncode."""
        validator.check_value_type('means', means, tuple, self.name)
        validator.check_value_type('stds', stds, tuple, self.name)
        for i, value in enumerate(means):
            validator.check_value_type("means[%d]" % i, value, [float], self.name)
        for i, value in enumerate(stds):
            validator.check_value_type("stds[%d]" % i, value, [float], self.name)
        validator.check_equal_int(len(means), 4, "means len", self.name)
        validator.check_equal_int(len(stds), 4, "stds len", self.name)


class BoundingBoxDecode(Primitive):
    """
    Decodes bounding boxes locations.

    The function of the operator is to calculate the offset, and this operator converts the offset into a Bbox,
    which is used to mark the target in the subsequent images, etc.

    Args:
        max_shape (tuple): The max size limit for decoding box calculation.
        means (tuple): The means of deltas calculation. Default: (0.0, 0.0, 0.0, 0.0).
        stds (tuple): The standard deviations of deltas calculation. Default: (1.0, 1.0, 1.0, 1.0).
        wh_ratio_clip (float): The limit of width and height ratio for decoding box calculation. Default: 0.016.

    Inputs:
        - **anchor_box** (Tensor) - Anchor boxes. The shape of `anchor_box` must be (n, 4).
        - **deltas** (Tensor) - Delta of boxes. Which has the same shape with `anchor_box`.

    Outputs:
        Tensor, decoded boxes. It has the same data type and shape as `anchor_box`.

    Raises:
        TypeError: If `means`, `stds` or `max_shape` is not a tuple.
        TypeError: If `wh_ratio_clip` is not a float.
        TypeError: If `anchor_box` or `deltas` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        """Initialize BoundingBoxDecode."""
        validator.check_value_type('means', means, tuple, self.name)
        validator.check_value_type('stds', stds, tuple, self.name)
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


class SampleDistortedBoundingBoxV2(Primitive):
    r"""
    Creates a single bounding box that is randomly distorted for an image.

    It is often used for object localization and image recognition tasks.
    In such tasks, bounding box annotations are supplied in addition to ground-truth
    labels, and data augmentation techniques are often used to randomly distort an image
    while preserving its content.

    This function takes the `image_size`, `bounding_boxes`, and
    a series of constraints as input, and outputs a randomly distorted localization of an
    object (i.e., bounding box) based on these inputs.

    The output is returned as 3 tensors:

    The output is returned as 3 tensors:
    `begin`, `size` and `bboxes`. The first 2 tensors can be fed directly
    into :class:`mindspore.ops.Slice` to crop the image.
    The latter is the generated distorted bounding box.

    Args:
        seed (int, optional): Random number seed. If either `seed` or `seed2` is set to a non-zero value,
            the seed is to the given value. Otherwise, a random seed is uesed. Default: 0.
        seed2 (int, optional): The second seed to avoid seed collision. Default: 0.
        aspect_ratio_range (Union[list(float), tuple(float)], optional): Specifying the valild range of aspect
            ratio of cropped area. Aspect ratio of area = area_width / area_height. The value of this
            attribute should be positive. Default: (0.75, 1.33).
        area_range (Union[list(float), tuple(float)], optional): The cropped area of the image must contain a
            fraction of the supplied image within this range. The value of this attribute should
            be in range (0.0, 1.0]. Default: (0.05, 1.0).
        max_attempts (int, optional): A poditive integer specifies the number of attempts that will be made to
            generate a cropped region of the image based on the given constraints. If the maximum number of
            attempts is exceeded without success, the function will return the entire original image.
            Default: 100.
        use_image_if_no_bounding_boxes (bool, optional): Controls behavior if no bounding boxes supplied.
            If no bounding boxes supplied (`bounding_boxes` in shape [0, N, 4] or [batch, 0, 4]), and this
            attribute is set True, then assume an implicit bounding box covering the
            whole input, else if this attribute is set False, then raise an error. Default: False.

    Inputs:
        - **image_size** (Tensor) - 1-D Tensor, containing [height, width, channels]. The value of this input
          tensor should be positive.
        - **bounding_boxes** (Tensor) - 3-D Tensor with shape :math:`(batch, N, 4)` describing the N
          bounding boxes associated with the image. The value of this input tensor should be in range [0.0, 1.0].
          The data type is float32.
        - **min_object_covered** (Tensor) - The least fraction of bounding box the croped area need to cover.
          This parameter's value should be between 0.0 and 1.0, inclusive. If the value is 0,
          the cropped area does not need to overlap with any of the supplied bounding boxes.
          The data type is float32.

    Outputs:
        - **begin** (Tensor) - A 1-D Tensor, containing [offset_height, offset_width, 0]. The data type is same as
          `image_size`.
        - **size** (Tensor) - A 1-D Tensor, containing [target_height, target_width, -1]. The data type is same as
          `image_size`. When the data type of `image_size` is uint8, the last value of `size`,
          which is originally -1, will be forced to 255.
        - **bboxes** (Tensor) - A 3-D Tensor with shape :math:`(1, 1, 4)`, containing
          the distorted bounding box. The data type is float32.

    Raises:
        TypeError: If `image_size` is not a Tensor.
        TypeError: If `bounding_boxes` is not a Tensor.
        TypeError: If `min_object_covered` is not a Tensor.
        TypeError: If `seed` or `seed2` is not an int.
        TypeError: If `aspect_ratio_range` is not a list or a tuple with type float.
        TypeError: If `area_range` is not a list or a tuple with type float.
        TypeError: If `use_image_if_no_bounding_boxes` is not a bool.
        ValueError: If the dimension of `image_size` is not 1.
        ValueError: If the elements of `image_size` is not 3.
        ValueError: If the dimension of `bounding_boxes` is not 3.
        ValueError: If the elements of each bounding box in `bounding_boxes` is not 4.
        ValueError: If the elements of `min_object_covered` is not 1.
        ValueError: If the elements of `aspect_ratio_range` list or tuple is not 2.
        ValueError: If the values of `aspect_ratio_range` is not positive.
        ValueError: If the second value of `aspect_ratio_range` is less than or equal to the first one.
        ValueError: If the elements of `area_range` list or tuple is not 2.
        ValueError: If the values of `area_range` is out of range (0.0, 1.0].
        ValueError: If the second value of `area_range` is less than or equal to the first one.
        ValueError: If the value of `max_attempts` is not positive int.
        ValueError: If `use_image_if_no_bounding_boxes` is False and no bounding boxes supplied.
        RuntimeError: If the values of `image_size` is not positive.
        RuntimeError: If the values of `bounding_boxes` is out of range [0.0, 1.0].
        RuntimeError: If the `bounding_boxes` cannot make up bounding box.
        RuntimeError: If the value of `min_object_covered` is out of range [0.0, 1.0].

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> image_size = Tensor([640, 480, 3], mindspore.int32)
        >>> bounding_boxes = Tensor([[[0.38, 0.17, 0.95, 0.40]]], mindspore.float32)
        >>> min_object_covered = Tensor([0.8], mindspore.float32)
        >>> sample_distorted_bounding_box_v2 = \
        ...   ops.SampleDistortedBoundingBoxV2(seed=1, seed2=1, aspect_ratio_range=(0.9, 1.1),
        ...                                    area_range=(0.1,1.0), max_attempts=100,
        ...                                    use_image_if_no_bounding_boxes=False)
        >>> output = sample_distorted_bounding_box_v2(image_size, bounding_boxes, min_object_covered)
        >>> begin, size, bboxes = output[0], output[1], output[2]
        >>> print(begin)
        [133   1   0]
        >>> print(size)
        [502 457  -1]
        >>> print(bboxes)
        [[[0.2078125  0.00208333 0.9921875  0.95416665]]]
    """

    @prim_attr_register
    def __init__(self, seed=0, seed2=0, \
                  aspect_ratio_range=(0.75, 1.33), \
                  area_range=(0.05, 1.0), \
                  max_attempts=100, \
                  use_image_if_no_bounding_boxes=False):
        validator.check_is_int(seed, "seed", self.name)
        validator.check_is_int(seed2, "seed2", self.name)
        validator.check_value_type("aspect_ratio_range", aspect_ratio_range, [list, tuple], self.name)
        validator.check_value_type("area_range", area_range, [list, tuple], self.name)
        validator.check_positive_int(max_attempts, "max_attempts", self.name)
        validator.check_bool(use_image_if_no_bounding_boxes, "use_image_if_no_bounding_boxes", self.name)
        for i, value in enumerate(aspect_ratio_range):
            validator.check_value_type("aspect_ratio_range[%d]" % i, value, [float], self.name)
        for i, value in enumerate(area_range):
            validator.check_value_type("area_range[%d]" % i, value, [float], self.name)


class CheckValid(Primitive):
    """
    Checks bounding box.

    Checks whether the bounding boxes specified by `bboxes` is valid.
    Returns True if the box is within borders specified by `img_metas`, False if not.

    Inputs:
        - **bboxes** (Tensor) - Bounding boxes tensor with shape :math:`(N, 4)`. :math:`N` indicates the number of
          bounding boxes, the value "4" indicates "x0", "x1", "y0", and "y1". Data type must be float16 or float32.
        - **img_metas** (Tensor) - Raw image size information with the format of (height, width, ratio), specifying
          the valid boundary(height * ratio, width * ratio). Data type must be float16 or float32.

    Outputs:
        Tensor, with shape of :math:`(N,)` and dtype of bool, specifying whether the bounding boxes is in the image.
        "True" indicates valid, while "False" indicates invalid.

    Raises:
        TypeError: If `bboxes` or `img_metas` is not a Tensor.
        TypeError: If dtype of `bboxes` or `img_metas` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
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
        """Initialize CheckValid."""
        self.init_prim_io_names(inputs=['bboxes', 'img_metas'], outputs=['output'])


class IOU(Primitive):
    r"""
    Calculates intersection over union for boxes.

    Computes the intersection over union (IOU) or the intersection over foreground (IOF) based on the ground-truth and
    predicted regions.

    Refer to :func:`mindspore.ops.iou` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> iou = ops.IOU(mode='iou')
        >>> anchor_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
        >>> gt_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
        >>> output = iou(anchor_boxes, gt_boxes)
        >>> print(output.shape)
        (3, 3)
    """

    @prim_attr_register
    def __init__(self, mode='iou'):
        """Initialize IOU."""
        if mode not in {'iou', 'iof'}:
            raise KeyError(f"For '{self.name}', only 'iou' or 'iof' are supported, but got 'mode': {mode}.")
        self.init_prim_io_names(inputs=['anchor_boxes', 'gt_boxes'], outputs=['overlap'])


class Partial(Primitive):
    """
    Makes a partial function instance. Partial function can be used to derived specialized
    functions from general functions by fixing the value of certain number of arguments.

    Inputs:
        - **args** (Union[FunctionType, Tensor]) - The function and bind arguments.

    Outputs:
        FunctionType, partial function bound with arguments.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
        >>> def show_input(x, y, z):
        ...     return x, y, z
        >>> partial = ops.Partial()
        >>> partial_show_input = partial(show_input, Tensor(1))
        >>> output1 = partial_show_input(Tensor(2), Tensor(3))
        >>> print(output1)
        (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64,
         value= 3))
        >>> output2 = partial_show_input(Tensor(3), Tensor(4))
        >>> print(output2)
        (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 3), Tensor(shape=[], dtype=Int64,
         value= 4))
    """

    # Side effect will propagated from the first argument to return value.
    side_effect_propagate = 1

    @prim_attr_register
    def __init__(self):
        """Initialize Partial."""
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
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.softmax = ops.Softmax()
        ...         self.depend = ops.Depend()
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
        """Initialize Depend."""
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

    def __call__(self, *args):
        return args[0]


class StopGradient(Primitive):
    """
    StopGradient is used for eliminating the effect of a value on the gradient,
    such as truncating the gradient propagation from an output of a function.

    Refer to :func:`mindspore.ops.stop_gradient` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> def net(x, y):
        ...     out1 = ops.MatMul()(x, y)
        ...     out2 = ops.MatMul()(x, y)
        ...     out2 = ops.StopGradient()(out2)
        ...     return out1, out2
        ...
        >>> x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
        >>> y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
        >>> grad_fn = ops.grad(net)
        >>> output = grad_fn(x, y)
        >>> print(output)
        [[1.4100001 1.6       6.5999994]
         [1.4100001 1.6       6.5999994]]
    """

    @prim_attr_register
    def __init__(self):
        pass


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
        """Initialize ConfusionMatrix."""
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
        self.init_prim_io_names(inputs=['optim_inputs', 'optim_input_shapes'], outputs=['key'])
        self.add_prim_attr("side_effect_hidden", True)

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


class PyInterpret(Primitive):
    r"""
    Interpret Python expression.
    """

    @prim_attr_register
    def __init__(self):
        super(PyInterpret, self).__init__(self.__class__.__name__)
        self.add_prim_attr('side_effect_io', True)


class PyExecute(PrimitiveWithInfer):
    r"""
    Execute Python expression.
    """

    @prim_attr_register
    def __init__(self):
        super(PyExecute, self).__init__(self.__class__.__name__)
        self.add_prim_attr('side_effect_io', True)
        self.add_prim_attr("primitive_target", "CPU")

    def infer_shape(self, *args):
        logger.error("The function output are empty tuple. Add a placeholder instead. "
                     "Do not use it as it could be any uninitialized data.")
        return ((1,),)

    def infer_dtype(self, *args):
        logger.error("The function output are empty tuple. Add a placeholder instead. "
                     "Do not use it as it could be any uninitialized data.")
        return (mstype.int32,)


class PyFunc(PrimitiveWithInfer):
    r"""
    Execute Python function.

    `PyFunc` encapsulates Python functions as an operator which could be compiled into computation graph.
    Unlike normal operators, it cannot be exported to MindIR as it is executed in current Python context.
    As only the weights of the network is stored in the checkpoint, network include `PyFunc` could save
    checkpoint and load to the network again, but will lose any Python function state.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        fn (function): Python function which inputs and outputs should be Python built-in scalar or numpy ndarray.
        in_types (list[:class:`mindspore.dtype`]): The type of the inputs.
        in_shapes (list[tuple[int]]): The dimensionality of the inputs. An empty list represents a scalar, otherwise it
                                      represent a numpy array.
        out_types (list[:class:`mindspore.dtype`]): The type of the outputs.
        out_shapes (list[tuple[int]]): The dimensionality of the outputs. An empty list represents a scalar, otherwise
                                       it represent a numpy array.
        stateful (bool): Whether the function is stateful or not.
                         If True, the execution order is same with model definition.

    Inputs:
        - **input_x** (Union(tuple[Tensor], list[Tensor])) - The input tuple or list
          is made up of multiple tensors.

    Outputs:
        tuple[Tensor], execution results Python functions.

    Raises:
        TypeError: The Python function execution failed.
        TypeError: The attributes(in_types/in_shapes/out_types/out_shapes) are inconsistent with Python function
                   specifications.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> def func(x1, x2):
        ...     return x1 + x2
        >>> x1 = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> x2 = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> op = P.PyFunc(func, [x1.dtype, x2.dtype], [x1.shape, x2.shape], [x1.dtype], [x1.shape])
        >>> output = op((x1, x2))
        >>> print(output[0].asnumpy())
        [2. 4. 6.]
    """

    def __init__(self, fn, in_types, in_shapes, out_types, out_shapes, stateful=True):
        super(PyFunc, self).__init__(self.__class__.__name__)
        add_pyfunc(id(fn), fn)
        self.add_prim_attr('fn_id', id(fn))
        self.add_prim_attr('in_types', in_types)
        self.add_prim_attr('in_shapes', in_shapes)
        self.add_prim_attr('out_types', out_types)
        self.add_prim_attr('out_shapes', out_shapes)
        validator.check_value_type("in_types", in_types, [list, tuple], self.name)
        validator.check_value_type("in_shapes", in_shapes, [list, tuple], self.name)
        validator.check("in_types length", len(in_types), "in_shapes length", len(in_shapes), Rel.EQ, self.name)
        validator.check_value_type("out_types", out_types, [list, tuple], self.name)
        validator.check_value_type("out_shapes", out_shapes, [list, tuple], self.name)
        validator.check("out_types length", len(out_types), "out_shapes length", len(out_shapes), Rel.EQ, self.name)
        self.add_prim_attr("side_effect_io", stateful)
        self.add_prim_attr("primitive_target", "CPU")
        fake_output = False
        single_scalar_output = False
        if not out_types:
            fake_output = True
        elif not out_shapes:
            single_scalar_output = True
        self.add_prim_attr("fake_output", fake_output)
        self.add_prim_attr("single_scalar_output", single_scalar_output)

    def infer_shape(self, *args):
        if self.out_shapes:
            return tuple(self.out_shapes)

        logger.warning("The function output are empty tuple. Add a placeholder instead. "
                       "Do not use it as it could be any uninitialized data.")
        return ((1,),)

    def infer_dtype(self, *args):
        if self.out_shapes:
            return tuple(self.out_types)

        logger.warning("The function output are empty tuple. Add a placeholder instead. "
                       "Do not use it as it could be any uninitialized data.")
        return (mstype.int32,)

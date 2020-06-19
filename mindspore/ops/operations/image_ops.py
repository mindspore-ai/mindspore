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

"""image_ops"""
from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype
from ..primitive import PrimitiveWithInfer, prim_attr_register


class CropAndResize(PrimitiveWithInfer):
    """
    Extracts crops from the input image tensor and resizes them.

    Note:
        In case that the output shape depends on crop_size, the crop_size should be constant.

    Args:
        method (str):  	An optional string specifying the sampling method for resizing.
            It can be either "bilinear" or "nearest" and default to "bilinear"
        extrapolation_value (float): An optional float defaults to 0. Value used for extrapolation, when applicable.

    Inputs:
        - **x** (Tensor) - The input image must be a 4-D tensor of shape [batch, image_height, image_width, depth].
            Types allowed: int8, int16, int32, int64, float16, float32, float64, uint8, uint16.
        - **boxes** (Tensor) - A 2-D tensor of shape [num_boxes, 4].
            The i-th row of the tensor specifies the coordinates of a box in the box_ind[i] image
            and is specified in normalized coordinates [y1, x1, y2, x2]. A normalized coordinate value of y is mapped to
            the image coordinate at y * (image_height - 1), so as the [0, 1] interval of normalized image height is
            mapped to [0, image_height - 1] in image height coordinates. We do allow y1 > y2, in which case the sampled
            crop is an up-down flipped version of the original image. The width dimension is treated similarly.
            Normalized coordinates outside the [0, 1] range are allowed, in which case we use extrapolation_value to
            extrapolate the input image values. Types allowd: float32.
        - **box_index** (Tensor) - A 1-D tensor of shape [num_boxes] with int32 values in [0, batch).
            The value of box_ind[i] specifies the image that the i-th box refers to. Types allowd: int32.
        - **crop_size** (Tensor) - Only constant value is allowd. Types allowed: int32.
            A 1-D tensor of 2 elements, size = [crop_height, crop_width].
            All cropped image patches are resized to this size. The aspect ratio of the image content is not preserved.
            Both crop_height and crop_width need to be positive.
    Outputs:
        A 4-D tensor of shape [num_boxes, crop_height, crop_width, depth] with type: float32.

    Examples:
        >>> class CropAndResizeNet(nn.Cell):
        >>>     def __init__(self, crop_size):
        >>>         super(CropAndResizeNet, self).__init__()
        >>>         self.crop_and_resize = P.CropAndResize()
        >>>         self.crop_size = crop_size
        >>>     @ms_function
        >>>     def construct(self, x, boxes, box_index):
        >>>         return self.crop_and_resize(x, boxes, box_index, self.crop_size)
        >>>
        >>> BATCH_SIZE = 1
        >>> NUM_BOXES = 5
        >>> IMAGE_HEIGHT = 256
        >>> IMAGE_WIDTH = 256
        >>> CHANNELS = 3
        >>> image = np.random.normal(size=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS]).astype(np.float32)
        >>> boxes = np.random.uniform(shape=[NUM_BOXES, 4]).astype(np.float32)
        >>> box_index = np.random.uniform(shape=[NUM_BOXES], low=0, high=BATCH_SIZE).astype(np.int32)
        >>> crop_size = np.array([24, 24]).astype(np.int32)
        >>> crop_and_resize = CropAndResizeNet(crop_size=Tensor(crop_size))
        >>> output = crop_and_resize(Tensor(image), Tensor(boxes), Tensor(box_index))
        >>> print(output.asnumpy())
    """

    @prim_attr_register
    def __init__(self, method="bilinear", extrapolation_value=0.0):
        """init CropAndResize"""
        self.init_prim_io_names(inputs=['x', 'boxes', 'box_index', 'crop_size'], outputs=['y'])
        validator.check_value_type("method", method, [str], self.name)
        validator.check_string("method", method, ["bilinear", "nearest"], self.name)
        self.method = method
        validator.check_value_type("extrapolation_value", extrapolation_value, [float], self.name)
        self.extrapolation_value = extrapolation_value

    def __infer__(self, x, boxes, box_index, crop_size):
        # get shape
        x_shape = list(x['shape'])
        boxes_shape = list(boxes['shape'])
        box_index_shape = list(box_index['shape'])
        crop_size_shape = list(crop_size['shape'])
        # get value
        if crop_size['value'] is None:
            raise ValueError(f"For {self.name}, crop_size must be const.")
        crop_size_value = crop_size['value'].asnumpy()
        # get dtype
        x_dtype = x['dtype']
        boxes_dtype = boxes['dtype']
        box_index_dtype = box_index['dtype']
        crop_size_dtype = crop_size['dtype']
        # check dytpe
        validator.check_tensor_type_same({"x": x_dtype},
                                         [mstype.int8, mstype.int16, mstype.int32, mstype.int64, mstype.float16,
                                          mstype.float32, mstype.float64, mstype.uint8, mstype.uint16], self.name)
        validator.check_tensor_type_same({"boxes": boxes_dtype}, [mstype.float32], self.name)
        validator.check_tensor_type_same({"box_index": box_index_dtype}, [mstype.int32], self.name)
        validator.check_tensor_type_same({"crop_size": crop_size_dtype}, [mstype.int32], self.name)
        # check input shape rank
        validator.check("x rank", len(x_shape), "expected", 4, Rel.EQ, self.name)
        validator.check("boxes rank", len(boxes_shape), "expected", 2, Rel.EQ, self.name)
        validator.check("box_index rank", len(box_index_shape), "expected", 1, Rel.EQ, self.name)
        validator.check("crop_size rank", len(crop_size_shape), "expected", 1, Rel.EQ, self.name)

        validator.check("boxes dim_0", boxes_shape[0], "box_index dim_0", box_index_shape[0], Rel.EQ, self.name)
        validator.check("boxes dim_1", boxes_shape[1], "expected", 4, Rel.EQ, self.name)

        num_boxes = boxes_shape[0]
        crop_height = crop_size_value[0]
        crop_width = crop_size_value[1]
        depth = x_shape[3]
        return {'shape': (num_boxes, crop_height, crop_width, depth),
                'dtype': mstype.float32,
                'value': None}

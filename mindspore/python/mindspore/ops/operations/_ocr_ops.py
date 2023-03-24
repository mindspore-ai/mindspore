# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Inner OCR operators."""

from __future__ import absolute_import
from mindspore.ops.primitive import PrimitiveWithInfer, Primitive, prim_attr_register
from mindspore.common import dtype as mstype


class GetShape(PrimitiveWithInfer):
    """
    Return the shape of the input tensor.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> shape = ops.GetShape()
        >>> output = shape((input_x,))
        >>> print(output)
        [3 2 1]
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=["x"], outputs=["y"])

    def infer_shape(self, x):
        return (len(x[0]),)

    def infer_dtype(self, x):
        return mstype.int32


class StringLength(PrimitiveWithInfer):
    """
    Return the length of the input string.

    Examples:
        >>> input_x = Tensor(a = np.array([["ab", "cde"], ["fghi", "jklmn"]]), dtype=mstype.string)
        >>> length = ops.StringLength()
        >>> output = length(input_x)
        >>> print(output)
        [[2, 3]
         [4, 5]]
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=["x"], outputs=["y"])

    def infer_shape(self, x):
        return x

    def infer_dtype(self, x):
        return mstype.int32


class OCRDetectionPreHandle(PrimitiveWithInfer):
    r"""
    This operator is used to prehandle images in ocr detection.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        data_format (str): Data format of the tensor passed to operator, is 'NHWC' or 'NCHW'. Default: "NHWC"

    Inputs:
        - **img** (Tensor) - Origin img tensor which has a 3-D shape. The data type must be uint8.

    Outputs:
        Tuple of 3 Tensors, representing imgs after prehandle, scale of the height and scale of the width.

        - **resized_img** (Tensor) - Imgs after prehandle, has the same shape of `img` input.
          The data type must be uint8.
        - **h_scale** (Scalar) - Height scale. The data type must be float32.
        - **w_scale** (Scalar) - Width scale. The data type must be float32.
    """
    @prim_attr_register
    def __init__(self, data_format="NHWC"):
        self.data_format = data_format
        self.init_prim_io_names(inputs=['img'], outputs=['resized_img', 'h_scale', 'w_scale'])

    def infer_shape(self, img):
        if self.data_format == "NHWC":
            resize_shp = (-1, -1, 3)
        else:
            resize_shp = (3, -1, -1)
        return resize_shp, (), ()

    def infer_dtype(self, img):
        return mstype.uint8, mstype.float32, mstype.float32


class OCRFindContours(PrimitiveWithInfer):
    r"""
    Find contours of the images.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        value_mode (int): Binarization mode.
                         `0` represents using `0` and `1`.
                         `1` represents using `0` and `255`. Default: 0.

    Inputs:
        - **img** (Tensor) - Origin img tensor which has a 2-D shape. The data type must be uint8.

    Outputs:
        Tuple of 3 Tensors.

        - **polys_data** (Tensor) - Point data of every contours. The data type must be int32.
        - **polys_offset** (Tensor) - Offset of every contours. The data type must be int32.
        - **polys_size** (Tensor) - Size of every contours. The data type must be int32.
    """
    @prim_attr_register
    def __init__(self, value_mode=0):
        self.init_prim_io_names(inputs=['img'], outputs=['polys_data', 'polys_offset', 'polys_size'])

    def infer_shape(self, img):
        return (255,), (255,), (255,)

    def infer_dtype(self, img):
        return mstype.int32, mstype.int32, mstype.int32


class BatchDilatePolys(PrimitiveWithInfer):
    r"""
    Batch dilate polygons according to expand_scale.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **polys_data** (Tensor) - Point data of every contours. The data type must be int32.
        - **polys_offset** (Tensor) - Offset of every contours. The data type must be int32.
        - **polys_size** (Tensor) - Size of every contours. The data type must be int32.
        - **score** (Tensor) - Score of every point in the image. The data type must be float32.
        - **min_border** (Tensor) - Minimum width of each polygon. The data type must be int32.
        - **min_area_thr** (Tensor) - Minimum area of each polygon. The data type must be int32.
        - **score_thr** (Tensor) - Minimum confidence score of each polygon. The data type must be float32.
        - **expand_scale** (Tensor) - Polygon expansion multiple. The data type must be float32.

    Outputs:
        Tuple of 3 Tensors.

        - **dilated_polys_data** (Tensor) - Point data of every dilated contours. The data type must be int32.
        - **dilated_polys_offset** (Tensor) - Offset of every dilated contours. The data type must be int32.
        - **dilated_polys_size** (Tensor) - Size of every dilated contours. The data type must be int32.
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['polys_data', 'polys_offset', 'polys_size', 'score',
                                        'min_border', 'min_area_thr', 'score_thr', 'expand_scale'],
                                outputs=['dilated_polys_data', 'dilated_polys_offset', 'dilated_polys_size'])

    def infer_shape(self, polys_data, polys_offset, polys_size, score,
                    min_border, min_area_thr, score_thr, expand_scale):
        return (255,), (255,), (255,)

    def infer_dtype(self, polys_data, polys_offset, polys_size, score,
                    min_border, min_area_thr, score_thr, expand_scale):
        return mstype.int32, mstype.int32, mstype.int32


class ResizeAndClipPolys(PrimitiveWithInfer):
    r"""
    Resize the dilated polygons according to scales.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **polys_data** (Tensor) - Point data of every contours. The data type must be int32.
        - **polys_offset** (Tensor) - Offset of every contours. The data type must be int32.
        - **polys_size** (Tensor) - Size of every contours. The data type must be int32.
        - **h_scale** (Tensor) - Expand scale of the height. The data type must be float32.
        - **w_scale** (Tensor) - Expand scale of the width. The data type must be float32.
        - **img_h** (Tensor) - Height of original image. The data type must be int32.
        - **img_w** (Tensor) - Width of original image. The data type must be int32.

    Outputs:
        Tuple of 4 Tensors.

        - **clipped_polys_data** (Tensor) - Point data of every clipped contours. The data type must be int32.
        - **clipped_polys_offset** (Tensor) - Offset of every clipped contours. The data type must be int32.
        - **clipped_polys_size** (Tensor) - Size of every clipped contours. The data type must be int32.
        - **clipped_polys_num** (Tensor) - Number of clipped polys. The data type must be int32.
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['polys_data', 'polys_offset', 'polys_size',
                                        'h_scale', 'w_scale', 'img_h', 'img_w'],
                                outputs=['clipped_polys_data', 'clipped_polys_offset',
                                         'clipped_polys_size', 'clipped_polys_num'])

    def infer_shape(self, polys_data, polys_offset, polys_size,
                    h_scale, w_scale, img_h, img_w):
        return (255,), (255,), (255,), ()

    def infer_dtype(self, polys_data, polys_offset, polys_size,
                    h_scale, w_scale, img_h, img_w):
        return mstype.int32, mstype.int32, mstype.int32, mstype.int32


class OCRDetectionPostHandle(PrimitiveWithInfer):
    r"""
    This operator is used to posthandle images in ocr detection.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **img** (Tensor) - Original image tensor. The data type must be uint8.
        - **polys_data** (Tensor) - Point data of every contours. The data type must be int32.
        - **polys_offset** (Tensor) - Offset of every contours. The data type must be int32.
        - **polys_size** (Tensor) - Size of every contours. The data type must be int32.

    Outputs:
        Tuple of 4 Tensors.

        - **imgs_data** (Tensor) - Image data of the origin image. The data type must be uint8.
        - **imgs_offset** (Tensor) - Offset of every imgs data. The data type must be int32.
        - **imgs_size** (Tensor) - Shape of every imgs data. The data type must be int32.
        - **rect_points** (Tensor) - Rect points of every imgs. The data type must be int32.
    """
    @prim_attr_register
    def __init__(self, data_format="NHWC"):
        self.init_prim_io_names(inputs=['img', 'polys_data', 'polys_offset', 'polys_size'],
                                outputs=['imgs_data', 'imgs_offset', 'imgs_size', 'rect_points'])

    def infer_shape(self, img, polys_data, polys_offset, polys_size):
        return (-1,), (-1,), (-1,), (-1,)

    def infer_dtype(self, img, polys_data, polys_offset, polys_size):
        return mstype.uint8, mstype.int32, mstype.int32, mstype.int32


class OCRIdentifyPreHandle(PrimitiveWithInfer):
    r"""
    This operator is used to prehandle images in ocr identification.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        size (Union[tuple[int], list[int]]): Image size after prehandle.
        data_format (str): Data format of the tensor passed to operator, is 'NHWC' or 'NCHW'.
                           Default: "NHWC"

    Inputs:
        - **imgs_data** (Tensor) - Image data of the origin image. The data type must be uint8.
        - **imgs_offset** (Tensor) - Offset of every imgs data. The data type must be int32.
        - **imgs_size** (Tensor) - Shape of every imgs data. The data type must be int32.

    Outputs:
        - **resized_img** (Tensor) - Imgs after prehandle. The data type must be uint8.
    """
    @prim_attr_register
    def __init__(self, size, data_format="NHWC"):
        self.init_prim_io_names(inputs=['imgs_data', 'imgs_offset', 'imgs_size'], outputs=['resized_imgs'])
        self.size = size
        self.data_format = data_format

    def infer_shape(self, imgs_data, imgs_offset, imgs_size):
        shp = (2, 3, self.size[0], self.size[1])
        if self.data_format == "NHWC":
            shp = (2, self.size[0], self.size[1], 3)
        return shp

    def infer_dtype(self, imgs_data, imgs_offset, imgs_size):
        return mstype.uint8


class OCRRecognitionPreHandle(Primitive):
    r"""
    This operator is used to prehandle images in ocr recognition.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        batch_size (int): Image batch size of each partition.
        data_format (str): Data format of the tensor passed to operator, is 'NHWC' or 'NCHW'. Default: "NHWC"
        pad_mode (str): Pad mode in prehandle. is 'REPLICATE' or 'ZERO'. Default: "REPLICATE"
        _op_max_shape (str): Max shape of each output. Default: "512,64,512,3;512;512;512"

    Inputs:
        - **imgs_data** (Tensor) - Image data of the origin image. The data type must be uint8.
        - **imgs_offset** (Tensor) - Offset of every imgs data. The data type must be int32.
        - **imgs_size** (Tensor) - Shape of every imgs data. The data type must be int32.

    Outputs:
        - **resized_img** (Tensor) - Imgs after prehandle. The data type must be uint8.
    """
    @prim_attr_register
    def __init__(self, batch_size, data_format="NHWC", pad_mode="REPLICATE",
                 _op_max_shape="512,64,512,3;512;512;512"):
        self.init_prim_io_names(inputs=['imgs_data', 'imgs_offset', 'imgs_size',
                                        'langs', 'langs_score'],
                                outputs=['imgs', 'imgs_relation', 'imgs_lang', 'imgs_piece_fillers'])
        self.data_format = data_format
        self.batch_size = batch_size
        self.add_prim_attr("_op_max_shape", _op_max_shape)


class BatchEnqueue(PrimitiveWithInfer):
    r"""
    This operator is used to batch input x according to attr batch_size and enqueue.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        batch_size (int): Image batch size of each dequeue.
        queue_name (str): Queue name.
        pad_mode (str): Pad mode in prehandle. is 'REPLICATE' or 'ZERO'. Default: "REPLICATE"
        queue_depth (int): Max elements in the queue. Default: 100.

    Inputs:
        - **x** (Tensor) - Data passed to the queue.
        - **queue_id** (Tensor) - Queue id. The data type must be uint32.

    Outputs:
        - **enqueue_count** (Tensor) - The count of the elements in the queue after enqueue.
    """
    @prim_attr_register
    def __init__(self, batch_size, queue_name, pad_mode="REPLICATE", queue_depth=100):
        self.init_prim_io_names(inputs=['x'], outputs=['enqueue_count'])
        self.add_prim_attr("side_effect_io", True)

    def infer_shape(self, x, q_id):
        return ()

    def infer_dtype(self, x, qiid):
        return mstype.int64


class Dequeue(PrimitiveWithInfer):
    r"""
    This operator is used to dequeue data according to queue_id.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        output_type (mstype): Data output type.
        output_shape (Union[tuple[int], list[int]]): Data output shape.
        queue_name (str): Queue name.

    Inputs:
        - **queue_id** (Tensor) - Queue id. The data type must be uint32.

    Outputs:
        - **data** (Tensor) - Dequeue data.
    """
    @prim_attr_register
    def __init__(self, output_type, output_shape, queue_name):
        self.init_prim_io_names(inputs=['queue_id'], outputs=['data'])
        self.output_type = output_type
        self.output_shape = output_shape
        self.add_prim_attr("side_effect_io", True)

    def infer_shape(self, q_id):
        return self.output_shape

    def infer_dtype(self, q_id):
        return self.output_type

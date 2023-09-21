# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Validators for image processing operations.
"""
import numbers
from functools import wraps
import numpy as np

from mindspore._c_dataengine import TensorOp, TensorOperation
from mindspore._c_expression import typing
from mindspore.dataset.core.validator_helpers import check_value, check_uint8, FLOAT_MIN_INTEGER, FLOAT_MAX_INTEGER, \
    check_pos_float32, check_float32, check_2tuple, check_range, check_positive, INT32_MAX, INT32_MIN, \
    parse_user_args, type_check, type_check_list, check_c_tensor_op, UINT8_MAX, UINT8_MIN, check_value_normalize_std, \
    check_value_cutoff, check_value_ratio, check_odd, check_non_negative_float32, check_non_negative_int32, \
    check_pos_int32, check_int32, check_tensor_op, deprecator_factory, check_valid_str
from mindspore.dataset.transforms.validators import check_transform_op_type
from .utils import Inter, Border, ImageBatchFormat, ConvertMode, SliceMode, AutoAugmentPolicy


def check_affine(method):
    """Wrapper method to check the parameters of Affine."""
    @wraps(method)
    def new_method(self, *args, **kwargs):
        [degrees, translate, scale, shear, resample, fill_value], _ = parse_user_args(method, *args, **kwargs)

        type_check(degrees, (int, float), "degrees")
        check_degrees(degrees)

        type_check(translate, (list, tuple), "translate")
        if len(translate) != 2:
            raise TypeError("The length of translate should be 2.")
        for i, t in enumerate(translate):
            type_check(t, (int, float), "translate[{}]".format(i))
            check_value(t, [-1.0, 1.0], "translate[{}]".format(i))

        type_check(scale, (int, float), "scale")
        check_positive(scale, "scale")

        type_check(shear, (numbers.Number, tuple, list), "shear")
        if isinstance(shear, (list, tuple)):
            if len(shear) != 2:
                raise TypeError("The length of shear should be 2.")
            for i, _ in enumerate(shear):
                type_check(shear[i], (int, float), "shear[{}]".format(i))

        type_check(resample, (Inter,), "resample")

        check_fill_value(fill_value)

        return method(self, *args, **kwargs)

    return new_method


def check_crop_size(size):
    """Wrapper method to check the parameters of crop size."""
    type_check(size, (int, list, tuple), "size")
    if isinstance(size, int):
        check_value(size, (1, FLOAT_MAX_INTEGER))
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        for index, value in enumerate(size):
            type_check(value, (int,), "size[{}]".format(index))
            check_value(value, (1, FLOAT_MAX_INTEGER))
    else:
        raise TypeError("Size should be a single integer or a list/tuple (h, w) of length 2.")


def check_crop_coordinates(coordinates):
    """Wrapper method to check the parameters of crop size."""
    type_check(coordinates, (list, tuple), "coordinates")
    if isinstance(coordinates, (tuple, list)) and len(coordinates) == 2:
        for index, value in enumerate(coordinates):
            type_check(value, (int,), "coordinates[{}]".format(index))
            check_value(value, (0, INT32_MAX), "coordinates[{}]".format(index))
    else:
        raise TypeError("Coordinates should be a list/tuple (y, x) of length 2.")


def check_cut_mix_batch_c(method):
    """Wrapper method to check the parameters of CutMixBatch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [image_batch_format, alpha, prob], _ = parse_user_args(method, *args, **kwargs)
        type_check(image_batch_format, (ImageBatchFormat,), "image_batch_format")
        type_check(alpha, (int, float), "alpha")
        type_check(prob, (int, float), "prob")
        check_pos_float32(alpha)
        check_positive(alpha, "alpha")
        check_value(prob, [0, 1], "prob")
        return method(self, *args, **kwargs)

    return new_method


def check_resize_size(size):
    """Wrapper method to check the parameters of resize."""
    if isinstance(size, int):
        check_value(size, (1, FLOAT_MAX_INTEGER))
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        for i, value in enumerate(size):
            type_check(value, (int,), "size at dim {0}".format(i))
            check_value(value, (1, INT32_MAX), "size at dim {0}".format(i))
    else:
        raise TypeError("Size should be a single integer or a list/tuple (h, w) of length 2.")


def check_mix_up_batch_c(method):
    """Wrapper method to check the parameters of MixUpBatch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [alpha], _ = parse_user_args(method, *args, **kwargs)
        type_check(alpha, (int, float), "alpha")
        check_positive(alpha, "alpha")
        check_pos_float32(alpha)

        return method(self, *args, **kwargs)

    return new_method


def check_normalize_param(mean, std):
    """Check the parameters of Normalize and NormalizePad operations."""
    type_check(mean, (list, tuple), "mean")
    type_check(std, (list, tuple), "std")
    if len(mean) != len(std):
        raise ValueError("Length of mean and std must be equal.")
    for i, mean_value in enumerate(mean):
        type_check(mean_value, (int, float), "mean[{}]".format(i))
        check_value(mean_value, [0, 255], "mean[{}]".format(i))
    for j, std_value in enumerate(std):
        type_check(std_value, (int, float), "std[{}]".format(j))
        check_value_normalize_std(std_value, [0, 255], "std[{}]".format(j))


def check_normalize_c_param(mean, std):
    type_check(mean, (list, tuple), "mean")
    type_check(std, (list, tuple), "std")
    if len(mean) != len(std):
        raise ValueError("Length of mean and std must be equal.")
    for mean_value in mean:
        check_value(mean_value, [0, 255], "mean_value")
    for std_value in std:
        check_value_normalize_std(std_value, [0, 255], "std_value")


def check_normalize_py_param(mean, std):
    type_check(mean, (list, tuple), "mean")
    type_check(std, (list, tuple), "std")
    if len(mean) != len(std):
        raise ValueError("Length of mean and std must be equal.")
    for mean_value in mean:
        check_value(mean_value, [0., 1.], "mean_value")
    for std_value in std:
        check_value_normalize_std(std_value, [0., 1.], "std_value")


def check_fill_value(fill_value):
    if isinstance(fill_value, int):
        check_uint8(fill_value, "fill_value")
    elif isinstance(fill_value, tuple) and len(fill_value) == 3:
        for i, value in enumerate(fill_value):
            check_uint8(value, "fill_value[{0}]".format(i))
    else:
        raise TypeError("fill_value should be a single integer or a 3-tuple.")


def check_padding(padding):
    """Parsing the padding arguments and check if it is legal."""
    type_check(padding, (tuple, list, numbers.Number), "padding")
    if isinstance(padding, numbers.Number):
        check_value(padding, (0, INT32_MAX), "padding")
    if isinstance(padding, (tuple, list)):
        if len(padding) not in (2, 4):
            raise ValueError("The size of the padding list or tuple should be 2 or 4.")
        for i, pad_value in enumerate(padding):
            type_check(pad_value, (int,), "padding[{}]".format(i))
            check_value(pad_value, (0, INT32_MAX), "pad_value")


def check_degrees(degrees):
    """Check if the `degrees` is legal."""
    type_check(degrees, (int, float, list, tuple), "degrees")
    if isinstance(degrees, (int, float)):
        check_non_negative_float32(degrees, "degrees")
    elif isinstance(degrees, (list, tuple)):
        if len(degrees) == 2:
            type_check_list(degrees, (int, float), "degrees")
            for value in degrees:
                check_float32(value, "degrees")
            if degrees[0] > degrees[1]:
                raise ValueError("degrees should be in (min,max) format. Got (max,min).")
        else:
            raise TypeError("If degrees is a sequence, the length must be 2.")


def check_random_color_adjust_param(value, input_name, center=1, bound=(0, FLOAT_MAX_INTEGER), non_negative=True):
    """Check the parameters in random color adjust operation."""
    type_check(value, (numbers.Number, list, tuple), input_name)
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError("The input value of {} cannot be negative.".format(input_name))
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise TypeError("If {0} is a sequence, the length must be 2.".format(input_name))
        if value[0] > value[1]:
            raise ValueError("{0} value should be in (min,max) format. Got ({1}, {2}).".format(input_name,
                                                                                               value[0], value[1]))
        check_range(value, bound)


def check_erasing_value(value):
    if not (isinstance(value, (numbers.Number,)) or
            (isinstance(value, (str,)) and value == 'random') or
            (isinstance(value, (tuple, list)) and len(value) == 3)):
        raise ValueError("The value for erasing should be either a single value, "
                         "or a string 'random', or a sequence of 3 elements for RGB respectively.")


def check_crop(method):
    """A wrapper that wraps a parameter checker around the original function(crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [coordinates, size], _ = parse_user_args(method, *args, **kwargs)
        check_crop_coordinates(coordinates)
        check_crop_size(size)

        return method(self, *args, **kwargs)

    return new_method


def check_center_crop(method):
    """A wrapper that wraps a parameter checker around the original function(center crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size], _ = parse_user_args(method, *args, **kwargs)
        check_crop_size(size)

        return method(self, *args, **kwargs)

    return new_method


def check_five_crop(method):
    """A wrapper that wraps a parameter checker around the original function(five crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size], _ = parse_user_args(method, *args, **kwargs)
        check_crop_size(size)

        return method(self, *args, **kwargs)

    return new_method


def check_erase(method):
    """Wrapper method to check the parameters of erase operation."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [top, left, height, width, value, inplace], _ = parse_user_args(
            method, *args, **kwargs)
        check_non_negative_int32(top, "top")
        check_non_negative_int32(left, "left")
        check_pos_int32(height, "height")
        check_pos_int32(width, "width")
        type_check(inplace, (bool,), "inplace")
        check_fill_value(value)

        return method(self, *args, **kwargs)

    return new_method


def check_random_posterize(method):
    """A wrapper that wraps a parameter checker around the original function(posterize operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [bits], _ = parse_user_args(method, *args, **kwargs)
        if bits is not None:
            type_check(bits, (list, tuple, int), "bits")
        if isinstance(bits, int):
            check_value(bits, [1, 8])
        if isinstance(bits, (list, tuple)):
            if len(bits) != 2:
                raise TypeError("Size of bits should be a single integer or a list/tuple (min, max) of length 2.")
            for item in bits:
                check_uint8(item, "bits")
            # also checks if min <= max
            check_range(bits, [1, 8])
        return method(self, *args, **kwargs)

    return new_method


def check_posterize(method):
    """A wrapper that wraps a parameter checker around the original function(posterize operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [bits], _ = parse_user_args(method, *args, **kwargs)
        type_check(bits, (int,), "bits")
        check_value(bits, [0, 8], "bits")
        return method(self, *args, **kwargs)

    return new_method


def check_resize_interpolation(method):
    """A wrapper that wraps a parameter checker around the original function(resize interpolation operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size, interpolation], _ = parse_user_args(method, *args, **kwargs)
        if interpolation is None:
            raise KeyError("Interpolation should not be None")
        check_resize_size(size)
        type_check(interpolation, (Inter,), "interpolation")

        return method(self, *args, **kwargs)

    return new_method

def check_device_target(method):
    """A wrapper that wraps a parameter checker"""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [device_target], _ = parse_user_args(method, *args, **kwargs)
        check_valid_str(device_target, ["CPU", "Ascend"], "device_target")
        return method(self, *args, **kwargs)
    return new_method


def check_resized_crop(method):
    """A wrapper that wraps a parameter checker around the original function(ResizedCrop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [top, left, height, width, size, interpolation], _ = parse_user_args(method, *args, **kwargs)
        check_non_negative_int32(top, "top")
        check_non_negative_int32(left, "left")
        check_pos_int32(height, "height")
        check_pos_int32(width, "width")
        type_check(interpolation, (Inter,), "interpolation")
        check_crop_size(size)

        return method(self, *args, **kwargs)
    return new_method


def check_resize(method):
    """A wrapper that wraps a parameter checker around the original function(resize operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size], _ = parse_user_args(method, *args, **kwargs)
        check_resize_size(size)

        return method(self, *args, **kwargs)

    return new_method


def check_size_scale_ration_max_attempts_paras(size, scale, ratio, max_attempts):
    """Wrapper method to check the parameters of RandomCropDecodeResize."""

    check_crop_size(size)
    if scale is not None:
        type_check(scale, (tuple, list), "scale")
        if len(scale) != 2:
            raise TypeError("scale should be a list/tuple of length 2.")
        type_check_list(scale, (float, int), "scale")
        if scale[0] > scale[1]:
            raise ValueError("scale should be in (min,max) format. Got (max,min).")
        check_range(scale, [0, FLOAT_MAX_INTEGER])
        check_positive(scale[1], "scale[1]")
    if ratio is not None:
        type_check(ratio, (tuple, list), "ratio")
        if len(ratio) != 2:
            raise TypeError("ratio should be a list/tuple of length 2.")
        check_pos_float32(ratio[0], "ratio[0]")
        check_pos_float32(ratio[1], "ratio[1]")
        if ratio[0] > ratio[1]:
            raise ValueError("ratio should be in (min,max) format. Got (max,min).")
    if max_attempts is not None:
        check_pos_int32(max_attempts, "max_attempts")


def check_random_adjust_sharpness(method):
    """Wrapper method to check the parameters of RandomAdjustSharpness."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [degree, prob], _ = parse_user_args(method, *args, **kwargs)
        type_check(degree, (float, int), "degree")
        check_non_negative_float32(degree, "degree")
        type_check(prob, (float, int), "prob")
        check_value(prob, [0., 1.], "prob")

        return method(self, *args, **kwargs)

    return new_method


def check_random_resize_crop(method):
    """A wrapper that wraps a parameter checker around the original function(random resize crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size, scale, ratio, interpolation, max_attempts], _ = parse_user_args(method, *args, **kwargs)
        if interpolation is not None:
            type_check(interpolation, (Inter,), "interpolation")
        check_size_scale_ration_max_attempts_paras(size, scale, ratio, max_attempts)

        return method(self, *args, **kwargs)

    return new_method


def check_random_auto_contrast(method):
    """Wrapper method to check the parameters of Python RandomAutoContrast op."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [cutoff, ignore, prob], _ = parse_user_args(method, *args, **kwargs)
        type_check(cutoff, (int, float), "cutoff")
        check_value_cutoff(cutoff, [0, 50], "cutoff")
        if ignore is not None:
            type_check(ignore, (list, tuple, int), "ignore")
        if isinstance(ignore, int):
            check_value(ignore, [0, 255], "ignore")
        if isinstance(ignore, (list, tuple)):
            for item in ignore:
                type_check(item, (int,), "item")
                check_value(item, [0, 255], "ignore")
        type_check(prob, (float, int,), "prob")
        check_value(prob, [0., 1.], "prob")

        return method(self, *args, **kwargs)

    return new_method


def check_prob(method):
    """A wrapper that wraps a parameter checker (to confirm probability) around the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [prob], _ = parse_user_args(method, *args, **kwargs)
        type_check(prob, (float, int,), "prob")
        check_value(prob, [0., 1.], "prob")

        return method(self, *args, **kwargs)

    return new_method


def check_alpha(method):
    """A wrapper method to check alpha parameter in RandomLighting."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [alpha], _ = parse_user_args(method, *args, **kwargs)
        type_check(alpha, (float, int,), "alpha")
        check_non_negative_float32(alpha, "alpha")

        return method(self, *args, **kwargs)

    return new_method


def check_normalize(method):
    """A wrapper that wraps a parameter checker around the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [mean, std, is_hwc], _ = parse_user_args(method, *args, **kwargs)
        check_normalize_param(mean, std)
        type_check(is_hwc, (bool,), "is_hwc")
        return method(self, *args, **kwargs)

    return new_method


def check_normalize_py(method):
    """A wrapper that wraps a parameter checker around the original function(normalize operation written in Python)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [mean, std], _ = parse_user_args(method, *args, **kwargs)
        check_normalize_py_param(mean, std)

        return method(self, *args, **kwargs)

    return new_method


def check_normalize_c(method):
    """A wrapper that wraps a parameter checker around the original function(normalize operation written in C++)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [mean, std], _ = parse_user_args(method, *args, **kwargs)
        check_normalize_c_param(mean, std)

        return method(self, *args, **kwargs)

    return new_method


def check_normalizepad(method):
    """A wrapper that wraps a parameter checker around the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [mean, std, dtype, is_hwc], _ = parse_user_args(method, *args, **kwargs)
        check_normalize_param(mean, std)
        type_check(is_hwc, (bool,), "is_hwc")
        if not isinstance(dtype, str):
            raise TypeError("dtype should be string.")
        if dtype not in ["float32", "float16"]:
            raise ValueError("dtype only supports float32 or float16.")

        return method(self, *args, **kwargs)

    return new_method


def check_normalizepad_c(method):
    """A wrapper that wraps a parameter checker around the original function(normalizepad written in C++)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [mean, std, dtype], _ = parse_user_args(method, *args, **kwargs)
        check_normalize_c_param(mean, std)
        if not isinstance(dtype, str):
            raise TypeError("dtype should be string.")
        if dtype not in ["float32", "float16"]:
            raise ValueError("dtype only support float32 or float16.")

        return method(self, *args, **kwargs)

    return new_method


def check_normalizepad_py(method):
    """A wrapper that wraps a parameter checker around the original function(normalizepad written in Python)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [mean, std, dtype], _ = parse_user_args(method, *args, **kwargs)
        check_normalize_py_param(mean, std)
        if not isinstance(dtype, str):
            raise TypeError("dtype should be string.")
        if dtype not in ["float32", "float16"]:
            raise ValueError("dtype only support float32 or float16.")

        return method(self, *args, **kwargs)

    return new_method


def check_random_crop(method):
    """Wrapper method to check the parameters of random crop."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size, padding, pad_if_needed, fill_value, padding_mode], _ = parse_user_args(method, *args, **kwargs)
        check_crop_size(size)
        type_check(pad_if_needed, (bool,), "pad_if_needed")
        if padding is not None:
            check_padding(padding)
        if fill_value is not None:
            check_fill_value(fill_value)
        if padding_mode is not None:
            type_check(padding_mode, (Border,), "padding_mode")

        return method(self, *args, **kwargs)

    return new_method


def check_random_color_adjust(method):
    """Wrapper method to check the parameters of random color adjust."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [brightness, contrast, saturation, hue], _ = parse_user_args(method, *args, **kwargs)
        check_random_color_adjust_param(brightness, "brightness")
        check_random_color_adjust_param(contrast, "contrast")
        check_random_color_adjust_param(saturation, "saturation")
        check_random_color_adjust_param(hue, 'hue', center=0, bound=(-0.5, 0.5), non_negative=False)

        return method(self, *args, **kwargs)

    return new_method


def check_resample_expand_center_fill_value_params(resample, expand, center, fill_value):
    type_check(resample, (Inter,), "resample")
    type_check(expand, (bool,), "expand")
    if center is not None:
        check_2tuple(center, "center")
        for value in center:
            type_check(value, (int, float), "center")
            check_value(value, [INT32_MIN, INT32_MAX], "center")
    check_fill_value(fill_value)


def check_random_rotation(method):
    """Wrapper method to check the parameters of random rotation."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [degrees, resample, expand, center, fill_value], _ = parse_user_args(method, *args, **kwargs)
        check_degrees(degrees)
        check_resample_expand_center_fill_value_params(resample, expand, center, fill_value)

        return method(self, *args, **kwargs)

    return new_method


def check_rotate(method):
    """Wrapper method to check the parameters of rotate."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [degrees, resample, expand, center, fill_value], _ = parse_user_args(method, *args, **kwargs)
        type_check(degrees, (float, int), "degrees")
        check_float32(degrees, "degrees")
        check_resample_expand_center_fill_value_params(resample, expand, center, fill_value)

        return method(self, *args, **kwargs)

    return new_method


def check_ten_crop(method):
    """Wrapper method to check the parameters of crop."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size, use_vertical_flip], _ = parse_user_args(method, *args, **kwargs)
        check_crop_size(size)

        if use_vertical_flip is not None:
            type_check(use_vertical_flip, (bool,), "use_vertical_flip")

        return method(self, *args, **kwargs)

    return new_method


def check_num_channels(method):
    """Wrapper method to check the parameters of number of channels."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [num_output_channels], _ = parse_user_args(method, *args, **kwargs)
        type_check(num_output_channels, (int,), "num_output_channels")
        if num_output_channels not in (1, 3):
            raise ValueError("Number of channels of the output grayscale image"
                             "should be either 1 or 3. Got {0}.".format(num_output_channels))

        return method(self, *args, **kwargs)

    return new_method


def check_pad(method):
    """Wrapper method to check the parameters of random pad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [padding, fill_value, padding_mode], _ = parse_user_args(method, *args, **kwargs)
        check_padding(padding)
        check_fill_value(fill_value)
        type_check(padding_mode, (Border,), "padding_mode")

        return method(self, *args, **kwargs)

    return new_method


def check_pad_to_size(method):
    """Wrapper method to check the parameters of PadToSize."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size, offset, fill_value, padding_mode], _ = parse_user_args(method, *args, **kwargs)

        type_check(size, (int, list, tuple), "size")
        if isinstance(size, int):
            check_pos_int32(size, "size")
        else:
            if len(size) != 2:
                raise ValueError("The size must be a sequence of length 2.")
            for i, value in enumerate(size):
                check_pos_int32(value, "size{0}".format(i))

        if offset is not None:
            type_check(offset, (int, list, tuple), "offset")
            if isinstance(offset, int):
                check_non_negative_int32(offset, "offset")
            else:
                if len(offset) not in [0, 2]:
                    raise ValueError("The offset must be empty or a sequence of length 2.")
                for i, _ in enumerate(offset):
                    check_non_negative_int32(offset[i], "offset{0}".format(i))

        check_fill_value(fill_value)
        type_check(padding_mode, (Border,), "padding_mode")

        return method(self, *args, **kwargs)

    return new_method


def check_perspective(method):
    """Wrapper method to check the parameters of Perspective."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [start_points, end_points, interpolation], _ = parse_user_args(method, *args, **kwargs)

        type_check_list(start_points, (list, tuple), "start_points")
        type_check_list(end_points, (list, tuple), "end_points")

        if len(start_points) != 4:
            raise TypeError("start_points should be a list or tuple of length 4.")
        for i, element in enumerate(start_points):
            type_check(element, (list, tuple), "start_points[{}]".format(i))
            if len(start_points[i]) != 2:
                raise TypeError("start_points[{}] should be a list or tuple of length 2.".format(i))
            check_int32(element[0], "start_points[{}][0]".format(i))
            check_int32(element[1], "start_points[{}][1]".format(i))
        if len(end_points) != 4:
            raise TypeError("end_points should be a list or tuple of length 4.")
        for i, element in enumerate(end_points):
            type_check(element, (list, tuple), "end_points[{}]".format(i))
            if len(end_points[i]) != 2:
                raise TypeError("end_points[{}] should be a list or tuple of length 2.".format(i))
            check_int32(element[0], "end_points[{}][0]".format(i))
            check_int32(element[1], "end_points[{}][1]".format(i))

        type_check(interpolation, (Inter,), "interpolation")

        return method(self, *args, **kwargs)

    return new_method


def check_slice_patches(method):
    """Wrapper method to check the parameters of slice patches."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [num_height, num_width, slice_mode, fill_value], _ = parse_user_args(method, *args, **kwargs)
        if num_height is not None:
            type_check(num_height, (int,), "num_height")
            check_value(num_height, (1, INT32_MAX), "num_height")
        if num_width is not None:
            type_check(num_width, (int,), "num_width")
            check_value(num_width, (1, INT32_MAX), "num_width")
        if slice_mode is not None:
            type_check(slice_mode, (SliceMode,), "slice_mode")
        if fill_value is not None:
            type_check(fill_value, (int,), "fill_value")
            check_value(fill_value, [0, 255], "fill_value")
        return method(self, *args, **kwargs)

    return new_method


def check_random_perspective(method):
    """Wrapper method to check the parameters of random perspective."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [distortion_scale, prob, interpolation], _ = parse_user_args(method, *args, **kwargs)

        type_check(distortion_scale, (float,), "distortion_scale")
        type_check(prob, (float,), "prob")
        check_value(distortion_scale, [0., 1.], "distortion_scale")
        check_value(prob, [0., 1.], "prob")
        type_check(interpolation, (Inter,), "interpolation")

        return method(self, *args, **kwargs)

    return new_method


def check_mix_up(method):
    """Wrapper method to check the parameters of mix up."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [batch_size, alpha, is_single], _ = parse_user_args(method, *args, **kwargs)
        type_check(is_single, (bool,), "is_single")
        type_check(batch_size, (int,), "batch_size")
        type_check(alpha, (int, float), "alpha")
        check_value(batch_size, (1, FLOAT_MAX_INTEGER))
        check_positive(alpha, "alpha")
        return method(self, *args, **kwargs)

    return new_method


def check_rgb_to_bgr(method):
    """Wrapper method to check the parameters of rgb_to_bgr."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [is_hwc], _ = parse_user_args(method, *args, **kwargs)
        type_check(is_hwc, (bool,), "is_hwc")
        return method(self, *args, **kwargs)

    return new_method


def check_rgb_to_hsv(method):
    """Wrapper method to check the parameters of rgb_to_hsv."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [is_hwc], _ = parse_user_args(method, *args, **kwargs)
        type_check(is_hwc, (bool,), "is_hwc")
        return method(self, *args, **kwargs)

    return new_method


def check_hsv_to_rgb(method):
    """Wrapper method to check the parameters of hsv_to_rgb."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [is_hwc], _ = parse_user_args(method, *args, **kwargs)
        type_check(is_hwc, (bool,), "is_hwc")
        return method(self, *args, **kwargs)

    return new_method


def check_random_erasing(method):
    """Wrapper method to check the parameters of random erasing."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [prob, scale, ratio, value, inplace, max_attempts], _ = parse_user_args(method, *args, **kwargs)

        type_check(prob, (float, int,), "prob")
        type_check_list(scale, (float, int,), "scale")
        if len(scale) != 2:
            raise TypeError("scale should be a list or tuple of length 2.")
        type_check_list(ratio, (float, int,), "ratio")
        if len(ratio) != 2:
            raise TypeError("ratio should be a list or tuple of length 2.")
        type_check(value, (int, list, tuple, str), "value")
        type_check(inplace, (bool,), "inplace")
        type_check(max_attempts, (int,), "max_attempts")
        check_erasing_value(value)

        check_value(prob, [0., 1.], "prob")
        if scale[0] > scale[1]:
            raise ValueError("scale should be in (min,max) format. Got (max,min).")
        check_range(scale, [0, FLOAT_MAX_INTEGER])
        check_positive(scale[1], "scale[1]")
        if ratio[0] > ratio[1]:
            raise ValueError("ratio should be in (min,max) format. Got (max,min).")
        check_value_ratio(ratio[0], [0, FLOAT_MAX_INTEGER])
        check_value_ratio(ratio[1], [0, FLOAT_MAX_INTEGER])
        if isinstance(value, int):
            check_value(value, (0, 255))
        if isinstance(value, (list, tuple)):
            for item in value:
                type_check(item, (int,), "value")
                check_value(item, [0, 255], "value")
        check_value(max_attempts, (1, FLOAT_MAX_INTEGER))

        return method(self, *args, **kwargs)

    return new_method


def check_cutout_new(method):
    """Wrapper method to check the parameters of cutout operation."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [length, num_patches, is_hwc], _ = parse_user_args(method, *args, **kwargs)
        type_check(length, (int,), "length")
        type_check(num_patches, (int,), "num_patches")
        type_check(is_hwc, (bool,), "is_hwc")
        check_value(length, (1, FLOAT_MAX_INTEGER))
        check_value(num_patches, (1, FLOAT_MAX_INTEGER))

        return method(self, *args, **kwargs)

    return new_method


def check_cutout(method):
    """Wrapper method to check the parameters of cutout operation."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [length, num_patches], _ = parse_user_args(method, *args, **kwargs)
        type_check(length, (int,), "length")
        type_check(num_patches, (int,), "num_patches")
        check_value(length, (1, FLOAT_MAX_INTEGER))
        check_value(num_patches, (1, FLOAT_MAX_INTEGER))

        return method(self, *args, **kwargs)

    return new_method


def check_decode(method):
    """Wrapper method to check the parameters of decode operation."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [to_pil], _ = parse_user_args(method, *args, **kwargs)
        type_check(to_pil, (bool,), "to_pil")

        return method(self, *args, **kwargs)

    return new_method


def check_linear_transform(method):
    """Wrapper method to check the parameters of linear transform."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [transformation_matrix, mean_vector], _ = parse_user_args(method, *args, **kwargs)
        type_check(transformation_matrix, (np.ndarray,), "transformation_matrix")
        type_check(mean_vector, (np.ndarray,), "mean_vector")

        if transformation_matrix.shape[0] != transformation_matrix.shape[1]:
            raise ValueError("transformation_matrix should be a square matrix. "
                             "Got shape {} instead.".format(transformation_matrix.shape))
        if mean_vector.shape[0] != transformation_matrix.shape[0]:
            raise ValueError("mean_vector length {0} should match either one dimension of the square"
                             "transformation_matrix {1}.".format(mean_vector.shape[0], transformation_matrix.shape))

        return method(self, *args, **kwargs)

    return new_method


def check_random_affine(method):
    """Wrapper method to check the parameters of random affine."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [degrees, translate, scale, shear, resample, fill_value], _ = parse_user_args(method, *args, **kwargs)
        check_degrees(degrees)

        if translate is not None:
            type_check(translate, (list, tuple), "translate")
            type_check_list(translate, (int, float), "translate")
            if len(translate) != 2 and len(translate) != 4:
                raise TypeError("translate should be a list or tuple of length 2 or 4.")
            for i, t in enumerate(translate):
                check_value(t, [-1.0, 1.0], "translate at {0}".format(i))

        if scale is not None:
            type_check(scale, (tuple, list), "scale")
            type_check_list(scale, (int, float), "scale")
            if len(scale) == 2:
                if scale[0] > scale[1]:
                    raise ValueError("Input scale[1] must be equal to or greater than scale[0].")
                check_range(scale, [0, FLOAT_MAX_INTEGER])
                check_positive(scale[1], "scale[1]")
            else:
                raise TypeError("scale should be a list or tuple of length 2.")

        if shear is not None:
            type_check(shear, (numbers.Number, tuple, list), "shear")
            if isinstance(shear, numbers.Number):
                check_positive(shear, "shear")
            else:
                type_check_list(shear, (int, float), "shear")
                if len(shear) not in (2, 4):
                    raise TypeError("shear must be of length 2 or 4.")
                if len(shear) == 2 and shear[0] > shear[1]:
                    raise ValueError("Input shear[1] must be equal to or greater than shear[0]")
                if len(shear) == 4 and (shear[0] > shear[1] or shear[2] > shear[3]):
                    raise ValueError("Input shear[1] must be equal to or greater than shear[0] and "
                                     "shear[3] must be equal to or greater than shear[2].")

        type_check(resample, (Inter,), "resample")

        if fill_value is not None:
            check_fill_value(fill_value)

        return method(self, *args, **kwargs)

    return new_method


def check_rescale(method):
    """Wrapper method to check the parameters of rescale."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [rescale, shift], _ = parse_user_args(method, *args, **kwargs)
        type_check(rescale, (numbers.Number,), "rescale")
        type_check(shift, (numbers.Number,), "shift")
        check_float32(rescale, "rescale")
        check_float32(shift, "shift")

        return method(self, *args, **kwargs)

    return new_method


def check_uniform_augment_cpp(method):
    """Wrapper method to check the parameters of UniformAugment C++ op."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [transforms, num_ops], _ = parse_user_args(method, *args, **kwargs)
        type_check(num_ops, (int,), "num_ops")
        check_positive(num_ops, "num_ops")

        if num_ops > len(transforms):
            raise ValueError("num_ops is greater than transforms list size.")
        parsed_transforms = []
        for op in transforms:
            if op and getattr(op, 'parse', None):
                parsed_transforms.append(op.parse())
            else:
                parsed_transforms.append(op)
        type_check(parsed_transforms, (list, tuple,), "transforms")
        for index, arg in enumerate(parsed_transforms):
            if not isinstance(arg, (TensorOp, TensorOperation)):
                raise TypeError("Type of Transforms[{0}] must be c_transform, but got {1}".format(index, type(arg)))

        return method(self, *args, **kwargs)

    return new_method


def check_uniform_augment(method):
    """Wrapper method to check the parameters of UniformAugment Unified op."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [transforms, num_ops], _ = parse_user_args(method, *args, **kwargs)
        type_check(num_ops, (int,), "num_ops")
        check_positive(num_ops, "num_ops")

        if num_ops > len(transforms):
            raise ValueError("num_ops is greater than transforms list size.")

        type_check(transforms, (list, tuple,), "transforms list")
        if not transforms:
            raise ValueError("transforms list can not be empty.")
        for ind, op in enumerate(transforms):
            check_tensor_op(op, "transforms[{0}]".format(ind))
            check_transform_op_type(ind, op)

        return method(self, *args, **kwargs)

    return new_method


def check_bounding_box_augment_cpp(method):
    """Wrapper method to check the parameters of BoundingBoxAugment C++ op."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [transform, ratio], _ = parse_user_args(method, *args, **kwargs)
        type_check(ratio, (float, int), "ratio")
        check_value(ratio, [0., 1.], "ratio")
        if transform and getattr(transform, 'parse', None):
            transform = transform.parse()
        type_check(transform, (TensorOp, TensorOperation), "transform")
        return method(self, *args, **kwargs)

    return new_method


def check_adjust_brightness(method):
    """Wrapper method to check the parameters of AdjustBrightness ops (Python and C++)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [brightness_factor], _ = parse_user_args(method, *args, **kwargs)
        type_check(brightness_factor, (float, int), "brightness_factor")
        check_value(brightness_factor, (0, FLOAT_MAX_INTEGER), "brightness_factor")
        return method(self, *args, **kwargs)

    return new_method


def check_adjust_contrast(method):
    """Wrapper method to check the parameters of AdjustContrast ops (Python and C++)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [contrast_factor], _ = parse_user_args(method, *args, **kwargs)
        type_check(contrast_factor, (float, int), "contrast_factor")
        check_value(contrast_factor, (0, FLOAT_MAX_INTEGER), "contrast_factor")
        return method(self, *args, **kwargs)

    return new_method


def check_adjust_gamma(method):
    """Wrapper method to check the parameters of AdjustGamma ops (Python and C++)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [gamma, gain], _ = parse_user_args(method, *args, **kwargs)
        type_check(gamma, (float, int), "gamma")
        check_value(gamma, (0, FLOAT_MAX_INTEGER))
        if gain is not None:
            type_check(gain, (float, int), "gain")
            check_value(gain, (FLOAT_MIN_INTEGER, FLOAT_MAX_INTEGER))
        return method(self, *args, **kwargs)

    return new_method


def check_adjust_hue(method):
    """Wrapper method to check the parameters of AdjustHue ops (Python and C++)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [hue_factor], _ = parse_user_args(method, *args, **kwargs)
        type_check(hue_factor, (float, int), "hue_factor")
        check_value(hue_factor, (-0.5, 0.5), "hue_factor")
        return method(self, *args, **kwargs)

    return new_method


def check_adjust_saturation(method):
    """Wrapper method to check the parameters of AdjustSaturation ops (Python and C++)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [saturation_factor], _ = parse_user_args(method, *args, **kwargs)
        type_check(saturation_factor, (float, int), "saturation_factor")
        check_value(saturation_factor, (0, FLOAT_MAX_INTEGER))
        return method(self, *args, **kwargs)

    return new_method


def check_adjust_sharpness(method):
    """Wrapper method to check the parameters of AdjustSharpness ops (Python and C++)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sharpness_factor], _ = parse_user_args(method, *args, **kwargs)
        type_check(sharpness_factor, (float, int), "sharpness_factor")
        check_value(sharpness_factor, (0, FLOAT_MAX_INTEGER))
        return method(self, *args, **kwargs)

    return new_method


def check_auto_contrast(method):
    """Wrapper method to check the parameters of AutoContrast ops (Python and C++)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [cutoff, ignore], _ = parse_user_args(method, *args, **kwargs)
        type_check(cutoff, (int, float), "cutoff")
        check_value_cutoff(cutoff, [0, 50], "cutoff")
        if ignore is not None:
            type_check(ignore, (list, tuple, int), "ignore")
        if isinstance(ignore, int):
            check_value(ignore, [0, 255], "ignore")
        if isinstance(ignore, (list, tuple)):
            for item in ignore:
                type_check(item, (int,), "item")
                check_value(item, [0, 255], "ignore")
        return method(self, *args, **kwargs)

    return new_method


def check_uniform_augment_py(method):
    """Wrapper method to check the parameters of Python UniformAugment op."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [transforms, num_ops], _ = parse_user_args(method, *args, **kwargs)
        type_check(transforms, (list,), "transforms")

        if not transforms:
            raise ValueError("transforms list is empty.")

        for transform in transforms:
            if isinstance(transform, TensorOp):
                raise ValueError("transform list only accepts Python operations.")

        type_check(num_ops, (int,), "num_ops")
        check_positive(num_ops, "num_ops")
        if num_ops > len(transforms):
            raise ValueError("num_ops cannot be greater than the length of transforms list.")

        return method(self, *args, **kwargs)

    return new_method


def check_positive_degrees(method):
    """A wrapper method to check degrees parameter in RandomSharpness and RandomColor ops (Python and C++)"""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [degrees], _ = parse_user_args(method, *args, **kwargs)

        if degrees is not None:
            if not isinstance(degrees, (list, tuple)):
                raise TypeError("degrees must be either a tuple or a list.")
            type_check_list(degrees, (int, float), "degrees")
            if len(degrees) != 2:
                raise ValueError("degrees must be a sequence with length 2.")
            for degree in degrees:
                check_value(degree, (0, FLOAT_MAX_INTEGER))
            if degrees[0] > degrees[1]:
                raise ValueError("degrees should be in (min,max) format. Got (max,min).")

        return method(self, *args, **kwargs)

    return new_method


def check_random_select_subpolicy_op(method):
    """Wrapper method to check the parameters of RandomSelectSubpolicyOp."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [policy], _ = parse_user_args(method, *args, **kwargs)
        type_check(policy, (list,), "policy")
        if not policy:
            raise ValueError("policy can not be empty.")
        for sub_ind, sub in enumerate(policy):
            type_check(sub, (list,), "policy[{0}]".format([sub_ind]))
            if not sub:
                raise ValueError("policy[{0}] can not be empty.".format(sub_ind))
            for op_ind, tp in enumerate(sub):
                check_2tuple(tp, "policy[{0}][{1}]".format(sub_ind, op_ind))
                check_c_tensor_op(tp[0], "op of (op, prob) in policy[{0}][{1}]".format(sub_ind, op_ind))
                check_value(tp[1], (0, 1), "prob of (op, prob) policy[{0}][{1}]".format(sub_ind, op_ind))

        return method(self, *args, **kwargs)

    return new_method


def check_random_solarize(method):
    """Wrapper method to check the parameters of RandomSolarizeOp."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [threshold], _ = parse_user_args(method, *args, **kwargs)

        type_check(threshold, (tuple,), "threshold")
        type_check_list(threshold, (int,), "threshold")
        if len(threshold) != 2:
            raise ValueError("threshold must be a sequence of two numbers.")
        for element in threshold:
            check_value(element, (0, UINT8_MAX))
        if threshold[1] < threshold[0]:
            raise ValueError("threshold must be in min max format numbers.")

        return method(self, *args, **kwargs)

    return new_method


def check_gaussian_blur(method):
    """Wrapper method to check the parameters of GaussianBlur."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [kernel_size, sigma], _ = parse_user_args(method, *args, **kwargs)

        type_check(kernel_size, (int, list, tuple), "kernel_size")
        if isinstance(kernel_size, int):
            check_value(kernel_size, (1, FLOAT_MAX_INTEGER), "kernel_size")
            check_odd(kernel_size, "kernel_size")
        elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2:
            for index, value in enumerate(kernel_size):
                type_check(value, (int,), "kernel_size[{}]".format(index))
                check_value(value, (1, FLOAT_MAX_INTEGER), "kernel_size")
                check_odd(value, "kernel_size[{}]".format(index))
        else:
            raise TypeError(
                "Kernel size should be a single integer or a list/tuple (kernel_width, kernel_height) of length 2.")

        if sigma is not None:
            type_check(sigma, (numbers.Number, list, tuple), "sigma")
            if isinstance(sigma, numbers.Number):
                check_value(sigma, (0, FLOAT_MAX_INTEGER), "sigma")
            elif isinstance(sigma, (list, tuple)) and len(sigma) == 2:
                for index, value in enumerate(sigma):
                    type_check(value, (numbers.Number,), "size[{}]".format(index))
                    check_value(value, (0, FLOAT_MAX_INTEGER), "sigma")
            else:
                raise TypeError("Sigma should be a single number or a list/tuple of length 2 for width and height.")

        return method(self, *args, **kwargs)

    return new_method


def check_convert_color(method):
    """Wrapper method to check the parameters of convertcolor."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [convert_mode], _ = parse_user_args(method, *args, **kwargs)
        type_check(convert_mode, (ConvertMode,), "convert_mode")
        return method(self, *args, **kwargs)

    return new_method


def check_auto_augment(method):
    """Wrapper method to check the parameters of AutoAugment."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [policy, interpolation, fill_value], _ = parse_user_args(method, *args, **kwargs)

        type_check(policy, (AutoAugmentPolicy,), "policy")
        type_check(interpolation, (Inter,), "interpolation")
        check_fill_value(fill_value)
        return method(self, *args, **kwargs)

    return new_method


def check_to_tensor(method):
    """Wrapper method to check the parameters of ToTensor."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [output_type], _ = parse_user_args(method, *args, **kwargs)

        # Check if output_type is mindspore.dtype
        if isinstance(output_type, (typing.Type,)):
            return method(self, *args, **kwargs)

        # Special case: Check if output_type is None (which is invalid)
        if output_type is None:
            # Use type_check to raise error with descriptive error message
            type_check(output_type, (typing.Type, np.dtype,), "output_type")

        try:
            # Check if output_type can be converted to numpy type
            _ = np.dtype(output_type)
        except (TypeError, ValueError):
            # Use type_check to raise error with descriptive error message
            type_check(output_type, (typing.Type, np.dtype,), "output_type")

        return method(self, *args, **kwargs)

    return new_method


def deprecated_c_vision(substitute_name=None, substitute_module=None):
    """Decorator for version 1.8 deprecation warning for legacy mindspore.dataset.vision.c_transforms operation.

    Args:
        substitute_name (str, optional): The substitute name for deprecated operation.
        substitute_module (str, optional): The substitute module for deprecated operation.
    """
    return deprecator_factory("1.8", "mindspore.dataset.vision.c_transforms", "mindspore.dataset.vision",
                              substitute_name, substitute_module)


def deprecated_py_vision(substitute_name=None, substitute_module=None):
    """Decorator for version 1.8 deprecation warning for legacy mindspore.dataset.vision.py_transforms operation.

    Args:
        substitute_name (str, optional): The substitute name for deprecated operation.
        substitute_module (str, optional): The substitute module for deprecated operation.
    """
    return deprecator_factory("1.8", "mindspore.dataset.vision.py_transforms", "mindspore.dataset.vision",
                              substitute_name, substitute_module)


def check_solarize(method):
    """Wrapper method to check the parameters of SolarizeOp."""

    @wraps(method)
    def new_method(self, *args, **kwargs):

        [threshold], _ = parse_user_args(method, *args, **kwargs)
        type_check(threshold, (float, int, list, tuple), "threshold")
        if isinstance(threshold, (float, int)):
            threshold = (threshold, threshold)
        type_check_list(threshold, (float, int), "threshold")
        if len(threshold) != 2:
            raise TypeError("threshold must be a single number or sequence of two numbers.")
        for i, value in enumerate(threshold):
            check_value(value, (UINT8_MIN, UINT8_MAX), "threshold[{}]".format(i))
        if threshold[1] < threshold[0]:
            raise ValueError("threshold must be in order of (min, max).")

        return method(self, *args, **kwargs)

    return new_method


def check_trivial_augment_wide(method):
    """Wrapper method to check the parameters of TrivialAugmentWide."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [num_magnitude_bins, interpolation, fill_value], _ = parse_user_args(method, *args, **kwargs)
        type_check(num_magnitude_bins, (int,), "num_magnitude_bins")
        check_value(num_magnitude_bins, (2, FLOAT_MAX_INTEGER), "num_magnitude_bins")
        type_check(interpolation, (Inter,), "interpolation")
        check_fill_value(fill_value)
        return method(self, *args, **kwargs)

    return new_method


def check_rand_augment(method):
    """Wrapper method to check the parameters of RandAugment."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [num_ops, magnitude, num_magnitude_bins, interpolation, fill_value], _ = parse_user_args(method, *args,
                                                                                                 **kwargs)

        type_check(num_ops, (int,), "num_ops")
        check_value(num_ops, (0, FLOAT_MAX_INTEGER), "num_ops")
        type_check(num_magnitude_bins, (int,), "num_magnitude_bins")
        check_value(num_magnitude_bins, (2, FLOAT_MAX_INTEGER), "num_magnitude_bins")
        type_check(magnitude, (int,), "magnitude")
        check_value(magnitude, (0, num_magnitude_bins), "magnitude", right_open_interval=True)
        type_check(interpolation, (Inter,), "interpolation")
        check_fill_value(fill_value)
        return method(self, *args, **kwargs)

    return new_method

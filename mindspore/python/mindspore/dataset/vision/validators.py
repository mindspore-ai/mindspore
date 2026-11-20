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
from mindspore.dataset.core.validator_helpers import (
    check_value,
    check_uint8,
    FLOAT_MAX_INTEGER,
    check_pos_float32,
    check_float32,
    check_2tuple,
    check_range,
    check_positive,
    INT32_MAX,
    INT32_MIN,
    parse_user_args,
    type_check,
    type_check_list,
    check_value_normalize_std,
    check_non_negative_float32,
    check_pos_int32,
    check_valid_str,
)
from .utils import (
    Inter,
    Border,
)


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
        type_check(padding, (int,), "padding")
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


def check_center_crop(method):
    """A wrapper that wraps a parameter checker around the original function(center crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [size], _ = parse_user_args(method, *args, **kwargs)
        check_crop_size(size)

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


def check_prob(method):
    """A wrapper that wraps a parameter checker (to confirm probability) around the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [prob], _ = parse_user_args(method, *args, **kwargs)
        type_check(prob, (float, int,), "prob")
        check_value(prob, [0., 1.], "prob")

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


def check_rgb_to_bgr(method):
    """Wrapper method to check the parameters of rgb_to_bgr."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [is_hwc], _ = parse_user_args(method, *args, **kwargs)
        type_check(is_hwc, (bool,), "is_hwc")
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

# Copyright 2019 Huawei Technologies Co., Ltd
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
"""Validators for TensorOps.
"""
import numbers
from functools import wraps

from .utils import Inter, Border
from ...transforms.validators import check_pos_int32, check_pos_float32, check_value, check_uint8, FLOAT_MAX_INTEGER, \
    check_bool, check_2tuple, check_range, check_list, check_type, check_positive, INT32_MAX


def check_inter_mode(mode):
    if not isinstance(mode, Inter):
        raise ValueError("Invalid interpolation mode.")


def check_border_type(mode):
    if not isinstance(mode, Border):
        raise ValueError("Invalid padding mode.")


def check_crop_size(size):
    """Wrapper method to check the parameters of crop size."""
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        size = size
    else:
        raise TypeError("Size should be a single integer or a list/tuple (h, w) of length 2.")
    for value in size:
        check_pos_int32(value)
    return size


def check_resize_size(size):
    """Wrapper method to check the parameters of resize."""
    if isinstance(size, int):
        check_pos_int32(size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        for value in size:
            check_value(value, (1, INT32_MAX))
    else:
        raise TypeError("Size should be a single integer or a list/tuple (h, w) of length 2.")
    return size


def check_normalize_c_param(mean, std):
    if len(mean) != len(std):
        raise ValueError("Length of mean and std must be equal")
    for mean_value in mean:
        check_pos_float32(mean_value)
    for std_value in std:
        check_pos_float32(std_value)


def check_normalize_py_param(mean, std):
    if len(mean) != len(std):
        raise ValueError("Length of mean and std must be equal")
    for mean_value in mean:
        check_value(mean_value, [0., 1.])
    for std_value in std:
        check_value(std_value, [0., 1.])


def check_fill_value(fill_value):
    if isinstance(fill_value, int):
        check_uint8(fill_value)
    elif isinstance(fill_value, tuple) and len(fill_value) == 3:
        for value in fill_value:
            check_uint8(value)
    else:
        raise TypeError("fill_value should be a single integer or a 3-tuple.")
    return fill_value


def check_padding(padding):
    """Parsing the padding arguments and check if it is legal."""
    if isinstance(padding, numbers.Number):
        top = bottom = left = right = padding

    elif isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            left = right = padding[0]
            top = bottom = padding[1]
        elif len(padding) == 4:
            left = padding[0]
            top = padding[1]
            right = padding[2]
            bottom = padding[3]
        else:
            raise ValueError("The size of the padding list or tuple should be 2 or 4.")
    else:
        raise TypeError("Padding can be any of: a number, a tuple or list of size 2 or 4.")
    return left, top, right, bottom


def check_degrees(degrees):
    """Check if the degrees is legal."""
    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError("If degrees is a single number, it cannot be negative.")
        degrees = (-degrees, degrees)
    elif isinstance(degrees, (list, tuple)):
        if len(degrees) != 2:
            raise ValueError("If degrees is a sequence, the length must be 2.")
    else:
        raise TypeError("Degrees must be a single non-negative number or a sequence")
    return degrees


def check_random_color_adjust_param(value, input_name, center=1, bound=(0, FLOAT_MAX_INTEGER), non_negative=True):
    """Check the parameters in random color adjust operation."""
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError("The input value of {} cannot be negative.".format(input_name))
        # convert value into a range
        value = [center - value, center + value]
        if non_negative:
            value[0] = max(0, value[0])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError("Please check your value range of {} is valid and "
                             "within the bound {}".format(input_name, bound))
    else:
        raise TypeError("Input of {} should be either a single value, or a list/tuple of "
                        "length 2.".format(input_name))
    factor = (value[0], value[1])
    return factor


def check_erasing_value(value):
    if not (isinstance(value, (numbers.Number, str, bytes)) or
            (isinstance(value, (tuple, list)) and len(value) == 3)):
        raise ValueError("The value for erasing should be either a single value, "
                         "or a string 'random', or a sequence of 3 elements for RGB respectively.")


def check_crop(method):
    """A wrapper that wrap a parameter checker to the original function(crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        size = (list(args) + [None])[0]
        if "size" in kwargs:
            size = kwargs.get("size")
        if size is None:
            raise ValueError("size is not provided.")
        size = check_crop_size(size)
        kwargs["size"] = size

        return method(self, **kwargs)

    return new_method


def check_resize_interpolation(method):
    """A wrapper that wrap a parameter checker to the original function(resize interpolation operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 2 * [None])[:2]
        size, interpolation = args
        if "size" in kwargs:
            size = kwargs.get("size")
        if "interpolation" in kwargs:
            interpolation = kwargs.get("interpolation")

        if size is None:
            raise ValueError("size is not provided.")
        size = check_resize_size(size)
        kwargs["size"] = size

        if interpolation is not None:
            check_inter_mode(interpolation)
            kwargs["interpolation"] = interpolation

        return method(self, **kwargs)

    return new_method


def check_resize(method):
    """A wrapper that wrap a parameter checker to the original function(resize operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        size = (list(args) + [None])[0]
        if "size" in kwargs:
            size = kwargs.get("size")

        if size is None:
            raise ValueError("size is not provided.")
        size = check_resize_size(size)
        kwargs["size"] = size

        return method(self, **kwargs)

    return new_method


def check_random_resize_crop(method):
    """A wrapper that wrap a parameter checker to the original function(random resize crop operation)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 5 * [None])[:5]
        size, scale, ratio, interpolation, max_attempts = args
        if "size" in kwargs:
            size = kwargs.get("size")
        if "scale" in kwargs:
            scale = kwargs.get("scale")
        if "ratio" in kwargs:
            ratio = kwargs.get("ratio")
        if "interpolation" in kwargs:
            interpolation = kwargs.get("interpolation")
        if "max_attempts" in kwargs:
            max_attempts = kwargs.get("max_attempts")

        if size is None:
            raise ValueError("size is not provided.")
        size = check_crop_size(size)
        kwargs["size"] = size

        if scale is not None:
            check_range(scale, [0, FLOAT_MAX_INTEGER])
            kwargs["scale"] = scale
        if ratio is not None:
            check_range(ratio, [0, FLOAT_MAX_INTEGER])
            check_positive(ratio[0])
            kwargs["ratio"] = ratio
        if interpolation is not None:
            check_inter_mode(interpolation)
            kwargs["interpolation"] = interpolation
        if max_attempts is not None:
            check_pos_int32(max_attempts)
            kwargs["max_attempts"] = max_attempts

        return method(self, **kwargs)

    return new_method


def check_prob(method):
    """A wrapper that wrap a parameter checker(check the probability) to the original function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        prob = (list(args) + [None])[0]
        if "prob" in kwargs:
            prob = kwargs.get("prob")
        if prob is not None:
            check_value(prob, [0., 1.])
            kwargs["prob"] = prob

        return method(self, **kwargs)

    return new_method


def check_normalize_c(method):
    """A wrapper that wrap a parameter checker to the original function(normalize operation written in C++)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 2 * [None])[:2]
        mean, std = args
        if "mean" in kwargs:
            mean = kwargs.get("mean")
        if "std" in kwargs:
            std = kwargs.get("std")

        if mean is None:
            raise ValueError("mean is not provided.")
        if std is None:
            raise ValueError("std is not provided.")
        check_normalize_c_param(mean, std)
        kwargs["mean"] = mean
        kwargs["std"] = std

        return method(self, **kwargs)

    return new_method


def check_normalize_py(method):
    """A wrapper that wrap a parameter checker to the original function(normalize operation written in Python)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 2 * [None])[:2]
        mean, std = args
        if "mean" in kwargs:
            mean = kwargs.get("mean")
        if "std" in kwargs:
            std = kwargs.get("std")

        if mean is None:
            raise ValueError("mean is not provided.")
        if std is None:
            raise ValueError("std is not provided.")
        check_normalize_py_param(mean, std)
        kwargs["mean"] = mean
        kwargs["std"] = std

        return method(self, **kwargs)

    return new_method


def check_random_crop(method):
    """Wrapper method to check the parameters of random crop."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 5 * [None])[:5]
        size, padding, pad_if_needed, fill_value, padding_mode = args

        if "size" in kwargs:
            size = kwargs.get("size")
        if "padding" in kwargs:
            padding = kwargs.get("padding")
        if "fill_value" in kwargs:
            fill_value = kwargs.get("fill_value")
        if "padding_mode" in kwargs:
            padding_mode = kwargs.get("padding_mode")
        if "pad_if_needed" in kwargs:
            pad_if_needed = kwargs.get("pad_if_needed")

        if size is None:
            raise ValueError("size is not provided.")
        size = check_crop_size(size)
        kwargs["size"] = size

        if padding is not None:
            padding = check_padding(padding)
            kwargs["padding"] = padding
        if fill_value is not None:
            fill_value = check_fill_value(fill_value)
            kwargs["fill_value"] = fill_value
        if padding_mode is not None:
            check_border_type(padding_mode)
            kwargs["padding_mode"] = padding_mode
        if pad_if_needed is not None:
            kwargs["pad_if_needed"] = pad_if_needed

        return method(self, **kwargs)

    return new_method


def check_random_color_adjust(method):
    """Wrapper method to check the parameters of random color adjust."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 4 * [None])[:4]
        brightness, contrast, saturation, hue = args
        if "brightness" in kwargs:
            brightness = kwargs.get("brightness")
        if "contrast" in kwargs:
            contrast = kwargs.get("contrast")
        if "saturation" in kwargs:
            saturation = kwargs.get("saturation")
        if "hue" in kwargs:
            hue = kwargs.get("hue")

        if brightness is not None:
            kwargs["brightness"] = check_random_color_adjust_param(brightness, "brightness")
        if contrast is not None:
            kwargs["contrast"] = check_random_color_adjust_param(contrast, "contrast")
        if saturation is not None:
            kwargs["saturation"] = check_random_color_adjust_param(saturation, "saturation")
        if hue is not None:
            kwargs["hue"] = check_random_color_adjust_param(hue, 'hue', center=0, bound=(-0.5, 0.5), non_negative=False)

        return method(self, **kwargs)

    return new_method


def check_random_rotation(method):
    """Wrapper method to check the parameters of random rotation."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 5 * [None])[:5]
        degrees, resample, expand, center, fill_value = args
        if "degrees" in kwargs:
            degrees = kwargs.get("degrees")
        if "resample" in kwargs:
            resample = kwargs.get("resample")
        if "expand" in kwargs:
            expand = kwargs.get("expand")
        if "center" in kwargs:
            center = kwargs.get("center")
        if "fill_value" in kwargs:
            fill_value = kwargs.get("fill_value")

        if degrees is None:
            raise ValueError("degrees is not provided.")
        degrees = check_degrees(degrees)
        kwargs["degrees"] = degrees

        if resample is not None:
            check_inter_mode(resample)
            kwargs["resample"] = resample
        if expand is not None:
            check_bool(expand)
            kwargs["expand"] = expand
        if center is not None:
            check_2tuple(center)
            kwargs["center"] = center
        if fill_value is not None:
            fill_value = check_fill_value(fill_value)
            kwargs["fill_value"] = fill_value

        return method(self, **kwargs)

    return new_method


def check_transforms_list(method):
    """Wrapper method to check the parameters of transform list."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        transforms = (list(args) + [None])[0]
        if "transforms" in kwargs:
            transforms = kwargs.get("transforms")
        if transforms is None:
            raise ValueError("transforms is not provided.")

        check_list(transforms)
        kwargs["transforms"] = transforms

        return method(self, **kwargs)

    return new_method


def check_random_apply(method):
    """Wrapper method to check the parameters of random apply."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        transforms, prob = (list(args) + 2 * [None])[:2]
        if "transforms" in kwargs:
            transforms = kwargs.get("transforms")
        if transforms is None:
            raise ValueError("transforms is not provided.")
        check_list(transforms)
        kwargs["transforms"] = transforms

        if "prob" in kwargs:
            prob = kwargs.get("prob")
        if prob is not None:
            check_value(prob, [0., 1.])
            kwargs["prob"] = prob

        return method(self, **kwargs)

    return new_method


def check_ten_crop(method):
    """Wrapper method to check the parameters of crop."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 2 * [None])[:2]
        size, use_vertical_flip = args
        if "size" in kwargs:
            size = kwargs.get("size")
        if "use_vertical_flip" in kwargs:
            use_vertical_flip = kwargs.get("use_vertical_flip")

        if size is None:
            raise ValueError("size is not provided.")
        size = check_crop_size(size)
        kwargs["size"] = size

        if use_vertical_flip is not None:
            check_bool(use_vertical_flip)
            kwargs["use_vertical_flip"] = use_vertical_flip

        return method(self, **kwargs)

    return new_method


def check_num_channels(method):
    """Wrapper method to check the parameters of number of channels."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        num_output_channels = (list(args) + [None])[0]
        if "num_output_channels" in kwargs:
            num_output_channels = kwargs.get("num_output_channels")
        if num_output_channels is not None:
            if num_output_channels not in (1, 3):
                raise ValueError("Number of channels of the output grayscale image"
                                 "should be either 1 or 3. Got {0}".format(num_output_channels))
            kwargs["num_output_channels"] = num_output_channels

        return method(self, **kwargs)

    return new_method


def check_pad(method):
    """Wrapper method to check the parameters of random pad."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 3 * [None])[:3]
        padding, fill_value, padding_mode = args
        if "padding" in kwargs:
            padding = kwargs.get("padding")
        if "fill_value" in kwargs:
            fill_value = kwargs.get("fill_value")
        if "padding_mode" in kwargs:
            padding_mode = kwargs.get("padding_mode")

        if padding is None:
            raise ValueError("padding is not provided.")
        padding = check_padding(padding)
        kwargs["padding"] = padding

        if fill_value is not None:
            fill_value = check_fill_value(fill_value)
            kwargs["fill_value"] = fill_value
        if padding_mode is not None:
            check_border_type(padding_mode)
            kwargs["padding_mode"] = padding_mode

        return method(self, **kwargs)

    return new_method


def check_random_perspective(method):
    """Wrapper method to check the parameters of random perspective."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 3 * [None])[:3]
        distortion_scale, prob, interpolation = args
        if "distortion_scale" in kwargs:
            distortion_scale = kwargs.get("distortion_scale")
        if "prob" in kwargs:
            prob = kwargs.get("prob")
        if "interpolation" in kwargs:
            interpolation = kwargs.get("interpolation")

        if distortion_scale is not None:
            check_value(distortion_scale, [0., 1.])
            kwargs["distortion_scale"] = distortion_scale
        if prob is not None:
            check_value(prob, [0., 1.])
            kwargs["prob"] = prob
        if interpolation is not None:
            check_inter_mode(interpolation)
            kwargs["interpolation"] = interpolation

        return method(self, **kwargs)

    return new_method


def check_mix_up(method):
    """Wrapper method to check the parameters of mix up."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 3 * [None])[:3]
        batch_size, alpha, is_single = args
        if "batch_size" in kwargs:
            batch_size = kwargs.get("batch_size")
        if "alpha" in kwargs:
            alpha = kwargs.get("alpha")
        if "is_single" in kwargs:
            is_single = kwargs.get("is_single")

        if batch_size is None:
            raise ValueError("batch_size")
        check_pos_int32(batch_size)
        kwargs["batch_size"] = batch_size
        if alpha is None:
            raise ValueError("alpha")
        check_positive(alpha)
        kwargs["alpha"] = alpha
        if is_single is not None:
            check_type(is_single, bool)
            kwargs["is_single"] = is_single

        return method(self, **kwargs)

    return new_method


def check_random_erasing(method):
    """Wrapper method to check the parameters of random erasing."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 6 * [None])[:6]
        prob, scale, ratio, value, inplace, max_attempts = args
        if "prob" in kwargs:
            prob = kwargs.get("prob")
        if "scale" in kwargs:
            scale = kwargs.get("scale")
        if "ratio" in kwargs:
            ratio = kwargs.get("ratio")
        if "value" in kwargs:
            value = kwargs.get("value")
        if "inplace" in kwargs:
            inplace = kwargs.get("inplace")
        if "max_attempts" in kwargs:
            max_attempts = kwargs.get("max_attempts")

        if prob is not None:
            check_value(prob, [0., 1.])
            kwargs["prob"] = prob
        if scale is not None:
            check_range(scale, [0, FLOAT_MAX_INTEGER])
            kwargs["scale"] = scale
        if ratio is not None:
            check_range(ratio, [0, FLOAT_MAX_INTEGER])
            kwargs["ratio"] = ratio
        if value is not None:
            check_erasing_value(value)
            kwargs["value"] = value
        if inplace is not None:
            check_bool(inplace)
            kwargs["inplace"] = inplace
        if max_attempts is not None:
            check_pos_int32(max_attempts)
            kwargs["max_attempts"] = max_attempts

        return method(self, **kwargs)

    return new_method


def check_cutout(method):
    """Wrapper method to check the parameters of cutout operation."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 2 * [None])[:2]
        length, num_patches = args
        if "length" in kwargs:
            length = kwargs.get("length")
        if "num_patches" in kwargs:
            num_patches = kwargs.get("num_patches")

        if length is None:
            raise ValueError("length")
        check_pos_int32(length)
        kwargs["length"] = length

        if num_patches is not None:
            check_pos_int32(num_patches)
            kwargs["num_patches"] = num_patches

        return method(self, **kwargs)

    return new_method


def check_linear_transform(method):
    """Wrapper method to check the parameters of linear transform."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 2 * [None])[:2]
        transformation_matrix, mean_vector = args
        if "transformation_matrix" in kwargs:
            transformation_matrix = kwargs.get("transformation_matrix")
        if "mean_vector" in kwargs:
            mean_vector = kwargs.get("mean_vector")

        if transformation_matrix is None:
            raise ValueError("transformation_matrix is not provided.")
        if mean_vector is None:
            raise ValueError("mean_vector is not provided.")

        if transformation_matrix.shape[0] != transformation_matrix.shape[1]:
            raise ValueError("transformation_matrix should be a square matrix. "
                             "Got shape {} instead".format(transformation_matrix.shape))
        if mean_vector.shape[0] != transformation_matrix.shape[0]:
            raise ValueError("mean_vector length {0} should match either one dimension of the square"
                             "transformation_matrix {1}.".format(mean_vector.shape[0], transformation_matrix.shape))

        kwargs["transformation_matrix"] = transformation_matrix
        kwargs["mean_vector"] = mean_vector

        return method(self, **kwargs)

    return new_method


def check_random_affine(method):
    """Wrapper method to check the parameters of random affine."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        args = (list(args) + 6 * [None])[:6]
        degrees, translate, scale, shear, resample, fill_value = args
        if "degrees" in kwargs:
            degrees = kwargs.get("degrees")
        if "translate" in kwargs:
            translate = kwargs.get("translate")
        if "scale" in kwargs:
            scale = kwargs.get("scale")
        if "shear" in kwargs:
            shear = kwargs.get("shear")
        if "resample" in kwargs:
            resample = kwargs.get("resample")
        if "fill_value" in kwargs:
            fill_value = kwargs.get("fill_value")

        if degrees is None:
            raise ValueError("degrees is not provided.")
        degrees = check_degrees(degrees)
        kwargs["degrees"] = degrees

        if translate is not None:
            if isinstance(translate, (tuple, list)) and len(translate) == 2:
                for t in translate:
                    if t < 0.0 or t > 1.0:
                        raise ValueError("translation values should be between 0 and 1")
            else:
                raise TypeError("translate should be a list or tuple of length 2.")
            kwargs["translate"] = translate

        if scale is not None:
            if isinstance(scale, (tuple, list)) and len(scale) == 2:
                for s in scale:
                    if s <= 0:
                        raise ValueError("scale values should be positive")
            else:
                raise TypeError("scale should be a list or tuple of length 2.")
            kwargs["scale"] = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                shear = (-1 * shear, shear)
            elif isinstance(shear, (tuple, list)) and (len(shear) == 2 or len(shear) == 4):
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    shear = [s for s in shear]
            else:
                raise TypeError("shear should be a list or tuple and it must be of length 2 or 4.")
            kwargs["shear"] = shear

        if resample is not None:
            check_inter_mode(resample)
            kwargs["resample"] = resample
        if fill_value is not None:
            fill_value = check_fill_value(fill_value)
            kwargs["fill_value"] = fill_value

        return method(self, **kwargs)

    return new_method


def check_rescale(method):
    """Wrapper method to check the parameters of rescale."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        rescale, shift = (list(args) + 2 * [None])[:2]
        if "rescale" in kwargs:
            rescale = kwargs.get("rescale")
        if "shift" in kwargs:
            shift = kwargs.get("shift")

        if rescale is None:
            raise ValueError("rescale is not provided.")
        check_pos_float32(rescale)
        kwargs["rescale"] = rescale

        if shift is None:
            raise ValueError("shift is not provided.")
        if not isinstance(shift, numbers.Number):
            raise TypeError("shift is not a number.")
        kwargs["shift"] = shift

        return method(self, **kwargs)

    return new_method

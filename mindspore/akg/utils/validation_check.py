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

"""validation check functions"""
from functools import wraps, reduce
from akg.utils.format_transform import get_shape

MAX_DATA_SIZE = 2 ** 31

def check_input_type_dict(input_dict, input_key, input_name):
    """
    check input parameter type for new type: dict.

    Note:
        rule1: key of input_dict should be in the input_key
        rule2: type of input_dict[shape] should be in (list, tuple), if have shape
        rule3: type of input_dict[dtype] should be in (str), if have dtype

    Args:
        input_dict (dict): input_dict
        input_key (list or tuple): all input key list, the key of input must in input_key
        input_name (str): input param name, only used for error print

    Returns:
        None
    """
    def _check_input_type(input_key, input_type):
        if not isinstance(input_dict[input_key], input_type):
            raise RuntimeError(
                "the input parameter %s[%s] must be %s, while type of input is %s" %
                (input_name, input_key, input_type, type(input_dict[input_key])))

    for key in input_dict.keys():
        if key not in input_key:
            raise RuntimeError(
                "the input parameter %s must have arrt <%s>" %
                (input_name, key))

        # check shape's type of input_dict, if have shape
        if key == "shape":
            _check_input_type(key, (list, tuple))

        # check dtype's type of input_dict, if have dtype
        if key == "dtype":
            _check_input_type(key, (str,))


def check_input_type_list_tuple(inputs, expect):
    """check inputs by a list or tuple of expected types."""
    if not isinstance(inputs, expect[1][0]):
        raise RuntimeError("the input parameter %s must be (list, tuple), while"
                           " type of input is %s" % (expect[0], type(inputs)))
    for inp in inputs:
        if not isinstance(inp, expect[1][1]):
            raise RuntimeError("The element in parameter %s must be %s, while "
                               "type of input is %s" % (
                                   expect[0], expect[1][1], type(inp)))


def check_input_type(*type_args, **_type_kwargs):
    """check input parameter type."""
    def out_wrapper(func):
        """outer wrapper function."""
        formal_parameter = func.__code__.co_varnames
        formal_parameter_list = list(zip(formal_parameter, type_args))

        @wraps(func)
        def in_wrapper(*args, **kwargs):
            """inner wrapper function."""
            for i, arg_v in enumerate(args):
                # add for new input dict, if dict, will check shape and dtype
                if isinstance(arg_v, dict):
                    check_input_type_dict(arg_v, arg_v.keys(),
                                          formal_parameter_list[i][0])

                if isinstance(formal_parameter_list[i][1], tuple):
                    if isinstance(formal_parameter_list[i][1][0], tuple) \
                            and len(formal_parameter_list[i][1]) == 2:
                        check_input_type_list_tuple(arg_v, formal_parameter_list[i])
                        continue

                if not isinstance(arg_v, formal_parameter_list[i][1]):
                    raise RuntimeError("the %sth input parameter %s must be %s, "
                                       "while type of input is %s" % (str(i), formal_parameter_list[i][0],
                                                                      formal_parameter_list[i][1],
                                                                      type(arg_v)))
            for i in kwargs:
                for j in formal_parameter_list:
                    if i in j:
                        if not isinstance(kwargs[i], j[1]):
                            raise RuntimeError("the input parameter %s must be "
                                               "%s, while type of input is %s"
                                               "" % (i, j[1], type(kwargs[i])))
                        break
            return func(*args, **kwargs)

        return in_wrapper

    return out_wrapper


def shape_dtype_max_size_check(shape):
    """check validation of tensor's shape."""
    if shape:
        mul = int(reduce(lambda x, y: int(x) * int(y), shape))
        if mul > MAX_DATA_SIZE:
            error_msg = "*".join([str(sh) for sh in shape])
            raise RuntimeError("Invalid shape, data is {} bytes ({}), which "
                               "exceed max data size {} bytes"
                               .format(mul, error_msg, MAX_DATA_SIZE))


def check_shape(tensor, length=None, tensor_name=""):
    """The common check rule for placeholder data."""
    shape = get_shape(tensor)
    if not shape:
        raise RuntimeError("The ndim of input tensor {} must more than 0, "
                           "actual input is {}".format(tensor_name, len(shape)))

    for shape_v in shape:
        if not isinstance(shape_v, int) or shape_v <= 0:
            raise RuntimeError("The type of tensor {} axis value must be "
                               "positive int and value more than 0,"
                               "actual input is ({}) {}".
                               format(tensor_name, type(shape_v), shape_v))

    if length and len(shape) != length:
        raise ValueError('The length of {} should be {}, while actual length is {}'.
                         format(tensor_name, length, len(shape)))


def ops_dtype_check(dtype, args):
    """check validation of op's dtype."""
    expected_dtype = list()

    def _get_expect_dtype(expected_dtype, arg):
        if isinstance(arg, str):
            expected_dtype.append(arg)
        elif isinstance(arg, (list, tuple)):
            for t in arg:
                _get_expect_dtype(expected_dtype, t)
        else:
            raise TypeError("arg should be either a string, "
                            "or a list/tuple of string, "
                            "while current is {}".format(type(arg)))

    _get_expect_dtype(expected_dtype, args)

    if isinstance(dtype, (list, tuple)):
        checking_dtype = [d.lower() for d in dtype]
    elif isinstance(dtype, str):
        checking_dtype = [dtype.lower()]
    else:
        raise TypeError("dtype should be either a string or a tuple/list of string")
    error_msg = "Supported dtype: {}, while received dtype: {}"
    if not set(checking_dtype).issubset(set(expected_dtype)):
        raise RuntimeError(error_msg.format(expected_dtype, checking_dtype))


def reduce_axis_check(reduce_shape, reduce_axis):
    """check validation of reduce axis for certain reduce shape."""
    dim = len(reduce_shape)
    if dim == 1 and int(reduce_shape[0]) == 1:
        raise RuntimeError("Error, reduce shape is 1. Scalar is not supported "
                           "for reduction, please input a vector.")
    if isinstance(reduce_axis, int):
        if reduce_axis not in range(-dim, dim):
            raise RuntimeError("Reduce axis should be in range [%d. %d)"
                               "" % (-dim, dim))
    elif isinstance(reduce_axis, (tuple, list)):
        if len(reduce_axis) > len(reduce_shape):
            raise RuntimeError("Reduce axis list exceed reduce shape length: "
                               "%d vs %d, error" % (len(reduce_axis), len(reduce_shape)))
        processed_axis = []
        for axis in reduce_axis:
            processed_axis.append(int(axis + dim) if axis < 0 else int(axis))
        if len(set(processed_axis)) < len(processed_axis):
            raise RuntimeError("Reduce axis list contains %d duplicated element, please check"
                               % (len(processed_axis) - len(set(processed_axis))))
        for axis in processed_axis:
            if axis >= dim:
                raise RuntimeError("Invalid reduce axis, axis should less than %d" % dim)
    elif reduce_axis is not None:
        raise RuntimeError("axis should be a list, tuple or int.")


def elemwise_dtype_check(dtype_a, dtype_b, supported_type=None):
    """check validation of tensor's dtype for element-wise op."""
    if supported_type:
        ops_dtype_check(dtype_a, supported_type)
        ops_dtype_check(dtype_b, supported_type)
    if dtype_a.lower() != dtype_b.lower():
        raise RuntimeError("Element-wise operation needs same data type, while "
                           "current is %s vs %s" % (dtype_a.lower(), dtype_b.lower()))


def auto_broadcast_check(shape_a, shape_b):
    """automatic broadcast check."""
    shape_l = get_shape(shape_a)
    shape_r = get_shape(shape_b)

    if len(shape_l) <= len(shape_r):
        shape_short = shape_l
        shape_long = shape_r
    else:
        shape_short = shape_r
        shape_long = shape_l

    dim_diff = len(shape_long) - len(shape_short)
    for i in range(dim_diff):
        shape_short.insert(0, 1)
    for i, shp in enumerate(shape_short):
        if int(shp) != int(shape_long[i]) and 1 not in [int(shp), int(shape_long[i])]:
            raise RuntimeError("Invalid auto broadcast, dim %d should be 1 or equal, "
                               "while now is %d vs %d" % (i, shp, shape_long[i]))


def check_int_list(array, array_name):
    """check whether all the elements are integers."""
    for num in array:
        if not isinstance(num, int):
            raise RuntimeError("Type of value in %s should be int, but got type %s" % (array_name, type(num)))

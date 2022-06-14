# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
# ===========================================================================
"""GraphKernel expander utils"""
from abc import ABCMeta, abstractmethod
from mindspore._extends.graph_kernel.model import model_builder as builder
from mindspore._extends.graph_kernel.model.model import GraphKernelUnsupportedException as GKException


class Expander(metaclass=ABCMeta):
    """
    Expander is the base class of expanders.

    The method `_expand` should be overridden to implement the operator detail.
    """
    def __init__(self, expand_info):
        self.name = expand_info["name"]
        self.inputs = expand_info["input_desc"]
        self.outputs = expand_info["output_desc"]
        self.attrs = expand_info["attr"]
        self.processor = expand_info["process"]

    def run(self):
        """
        Expand the operator to a graph.

        `GraphKernelUnsupportedException` would be raised if check failed.
        """
        self._check()
        graph_builder = builder.GraphBuilder()
        with graph_builder.graph_scope(self.name) as graph_scope:
            # transform input_desc to Tensor
            self.inputs = [graph_builder.tensor(inp['shape'], inp['data_type'], inp['format']) for inp in self.inputs]
            graph_scope.set_input(*self.inputs)
            outputs = self._expand(graph_builder)
            if isinstance(outputs, (list, tuple)):
                self._check_output_same(outputs)
                graph_scope.set_output(*outputs)
            else:
                self._check_output_same([outputs])
                graph_scope.set_output(outputs)

        graph = graph_builder.get()[0]
        graph.set_processor(self.processor)
        return graph

    def _check(self):
        """Check inputs"""

    def _check_output_same(self, outputs):
        for index, value in enumerate(self.outputs):
            if list(outputs[index].shape) != list(value['shape']):
                raise GKException("{} 's output shape {} is wrong. Expected:{}".format(
                    self.__class__.__name__, list(outputs[index].shape), list(value['shape'])))
            if outputs[index].dtype != value['data_type']:
                raise GKException("{} 's output data_type {} is wrong. Expected: {}".format(
                    self.__class__.__name__, outputs[index].dtype, value['data_type']))
            if outputs[index].data_format != value['format']:
                raise GKException("{} 's output format {} is wrong. Expected: {}".format(
                    self.__class__.__name__, outputs[index].data_format, value['format']))

    @abstractmethod
    def _expand(self, graph_builder):
        """Expand operator, this function should be overridden in subclass"""
        raise Exception("_expand() is not implemented in {}".format(self.__class__.__name__))


class ExpanderInfoValidator:
    """ExpanderInfoValidator is the utility class which defines the validator decorator for expanders"""

    def __init__(self):
        """Init"""

    @staticmethod
    def _add_check_function(kls, func):
        """
        Rewrite the function `_check` in class Expander
        to append the new `func` after the original checks.
        """
        old_check = getattr(kls, "_check")

        def new_check(obj):
            old_check(obj)
            func(obj)

        setattr(kls, "_check", new_check)

    @staticmethod
    def add_format(*input_format):
        """
        Add new supported format for the operator

        this function will add a list `__supported_formats` into the expander,
        saving the whitelist of formats that this op supports.
        it also rewrites the `_check` function to check the formats.
        """
        format_list_name = "__supported_formats"

        def _check_format(obj):
            inp_formats = [inp['format'] for inp in obj.inputs]
            for formats in getattr(obj, format_list_name):
                if len(formats) != len(inp_formats):
                    raise GKException("For '{}', length of registered format is different from the length of inputs "
                                      "format: {} vs {}".format(obj.name, len(formats), len(inp_formats)))
                if all((fmt == inp for fmt, inp in zip(formats, inp_formats))):
                    return
            raise GKException("Unregistered format ({}) for op {}".format(','.join(inp_formats), obj.name))

        def wrapper(cls):
            if not issubclass(cls, Expander):
                raise Exception("{} should be subclass of Expander.".format(cls.__name__))
            if not hasattr(cls, format_list_name):
                setattr(cls, format_list_name, list())
                ExpanderInfoValidator._add_check_function(cls, _check_format)
            getattr(cls, format_list_name).append(input_format)
            return cls

        return wrapper

    @staticmethod
    def check_all_formats_same(kls):
        """Check that all formats are the same"""

        # Ensure no args case can return a class
        def _check(*args):
            def _check_format(obj):
                inp_formats = [inp['format'] for inp in obj.inputs]
                if all((fmt == inp_formats[0] for fmt in inp_formats[1:])):
                    return
                raise GKException("[check_all_formats_same] unmatched formats ({}) for op {}".format(
                    ','.join(inp_formats), obj.name))

            def wrapper(cls):
                if not issubclass(cls, Expander):
                    raise Exception("{} should be subclass of Expander.".format(cls.__name__))
                ExpanderInfoValidator._add_check_function(cls, _check_format)
                return cls

            return wrapper

        return _check()(kls)

    @staticmethod
    def check_attrs(*args):
        """Check the attrs exist"""

        def _check_attr(obj):
            for a in args:
                if a not in obj.attrs:
                    raise GKException("attr '{}' does not exist.".format(a))

        def wrapper(cls):
            if not issubclass(cls, Expander):
                raise Exception("{} should be subclass of Expander.".format(cls.__name__))
            ExpanderInfoValidator._add_check_function(cls, _check_attr)
            return cls

        return wrapper


def to_frac_z_axis(ori_shape, ori_axis):
    """
    judge the format is fractal NZ
    Parameters
    ----------
    ori_shape: list or tuple
        original shape of input
    ori_axis: list or tuple
        original axis of original shape to operate
    Returns
    -------
    output: list
        axis of the fractal Nz shape
    """
    frac_z_axis = list(ori_axis)
    shape_len = len(ori_shape)
    axis_count = len(frac_z_axis)
    axis_negative_1 = shape_len - 1
    axis_negative_2 = shape_len - 2
    for i in range(axis_count):
        axis_index = (frac_z_axis[i] + shape_len) % shape_len
        if axis_index == axis_negative_1:
            if frac_z_axis[i] > shape_len - 2:  # akg:[2,3] [1,4] tbe:[2,4] [1,3]
                frac_z_axis[i] = axis_index - 1
                frac_z_axis.append(axis_index + 2)
            else:  # no case cover this branch now
                frac_z_axis[i] = axis_index - 1
                frac_z_axis.append(axis_index + 2)
        elif axis_index == axis_negative_2:
            frac_z_axis[i] = axis_index + 1
            frac_z_axis.append(axis_index + 2)
        else:
            frac_z_axis[i] = axis_index
    return frac_z_axis


def infer_shape_from_fractalnz(fractal):
    "get original shape from fractalnz shape"
    shape = []
    dims = len(fractal)
    batch = dims - 4
    for i in range(batch):
        shape.append(fractal[i])
    m = fractal[dims - 3] * fractal[dims - 2]
    n = fractal[dims - 4] * fractal[dims - 1]
    shape.append(m)
    shape.append(n)
    return shape


def get_reduced_ori_shape(shape, axis):
    "get shape after reduced which is based on original shape"
    reduced_ori_shape = []
    for i, value in enumerate(shape):
        if i in axis:
            reduced_ori_shape.append(1)
        else:
            reduced_ori_shape.append(value)
    return reduced_ori_shape


def get_reduce_axis_shape(shape, data_format, axis):
    """
    Get the reduce axis under format `data_format` and original reduced shape.
    Parameters
    ----------
    shape: list or tuple
        shape of input
    data_format: str
        data format of input
    axis: None, int, list or tuple
        reduce axis of the original shape
    Returns
    -------
    reduce_axis: list
        reduce axis of the `data_format` shape
    ori_reduced_shape: list
        original reduced shape
    """
    ori_shape = shape
    if data_format == "FRACTAL_NZ":
        ori_shape = infer_shape_from_fractalnz(shape)
    if not axis:
        axis = []
        for i, _ in enumerate(ori_shape):
            axis.append(i)
    else:
        if isinstance(axis, int):
            axis = [axis]
        for i, _ in enumerate(list(axis)):
            if axis[i] < 0:
                axis[i] += len(ori_shape)

    ori_reduced_shape = get_reduced_ori_shape(ori_shape, axis)
    reduce_axis = axis
    if data_format == "FRACTAL_NZ":
        reduce_axis = to_frac_z_axis(ori_shape, axis)
    return reduce_axis, ori_reduced_shape

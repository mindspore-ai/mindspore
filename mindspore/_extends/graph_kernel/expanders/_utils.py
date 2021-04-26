# Copyright 2021 Huawei Technologies Co., Ltd
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


class Expander:
    """
    Expander is the base class of expanders.

    The method `_expand` should be overridden to implement the operator detail.
    """
    __metaclass__ = ABCMeta

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
                graph_scope.set_output(*outputs)
            else:
                graph_scope.set_output(outputs)

        graph = graph_builder.get()[0]
        graph.set_processor(self.processor)
        return graph

    def _check(self):
        """Check inputs"""

    @abstractmethod
    def _expand(self, graph_builder):
        """Expand operator, this function should be overridden in subclass"""
        raise Exception("_expand() is not implemented in {}".format(self.__class__.__name__))


class ExpanderInfoValidator:
    """ExpanderInfoValidator is the utility class which defines the validator decorator for expanders"""
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
                    raise GKException("length of registered format doesn't match with the input of {}".format(obj.name))
                if all([fmt == inp for fmt, inp in zip(formats, inp_formats)]):
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
        def _check_format(obj):
            inp_formats = [inp['format'] for inp in obj.inputs]
            if all([fmt == inp_formats[0] for fmt in inp_formats[1:]]):
                return
            raise GKException("[check_all_formats_same] unmatched formats ({}) for op {}".format(
                ','.join(inp_formats), obj.name))

        def wrapper(*args, **kargs):
            if not issubclass(kls, Expander):
                raise Exception("{} should be subclass of Expander.".format(kls.__name__))
            ExpanderInfoValidator._add_check_function(kls, _check_format)
            return kls(*args, **kargs)

        return wrapper

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

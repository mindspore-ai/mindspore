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
"""Patterns for describing graphs"""
from mindspore.ops import Primitive
from mindspore.common.tensor import Tensor
from mindspore._c_expression import Pattern, OneOf_, Prim_, Call_, NoneOf_, Any, NewTensor_, NewParameter_, Imm

__all__ = [
    "OneOf",
    "Prim",
    "Call",
    "NoneOf",
    "Any",
    "NewTensor",
    "NewParameter",
    "Imm"
]


class OneOf(OneOf_):
    r"""
    Express a pattern which allows a list of patterns.
    """
    def __init__(self, patterns=None):
        r"""
        Args:
            patterns(Union[:class:`mindspore.graph_utils.graph_pattern`,
                           tuple[:class:`mindspore.graph_utils.graph_pattern`],
                           list[:class:`mindspore.graph_utils.graph_pattern`]]): list of allowed patterns,
                each element should be one of the exposed Pattern instance.

        Raises:
            TypeError: raise type error for invalid inputs.
        """
        self.patterns = patterns
        if isinstance(patterns, Pattern):
            OneOf_.__init__(self, [patterns])
        elif isinstance(patterns, (tuple, list)) and all(isinstance(pattern, Pattern) for pattern in patterns):
            OneOf_.__init__(self, patterns)
        else:
            raise TypeError(f"Expect patterns to be a list of Patterns/Pattern, got : {patterns}")


class Prim(Prim_):
    r"""
    Express a pattern of certain primitive type(s).

    NOTE:
        This pattern will match and only match the primitive value node. If matching primitive CNode is needed,
        please refer to CallWith pattern.
    """
    def __init__(self, types, name=None):
        r"""
        Args:
            types (Union[str, :class:`mindspore.ops.Primitive`, list[:class:`mindspore.ops.Primitive`],
                   tuple[:class:`mindspore.ops.Primitive`]):
                Specify allowed types.
                If it is a string, the form could be
                    1) a single primitive type, e.g. 'Conv2D'
                    2) a set of primitive types separated by '|', e.g. 'MatMul|Conv2D'
                It can also be a Primitive or a list/tuple of Primitives, e.g. [ops.Conv2D(1, 6)]
            name (str): name of the pattern, optional. Default: None.

        Raises:
            TypeError: raise type error for invalid argument.
        """
        if name is not None and not isinstance(name, str):
            raise TypeError(f"Expect string, got : {name}")
        self.name = name
        if isinstance(types, str):
            if self.name is None:
                self.name = types
            self.types = types.split('|')
        elif isinstance(types, Primitive):
            if self.name is None:
                self.name = types.name
            self.types = [types]
        elif isinstance(types, (tuple, list)) and all(isinstance(tp, Primitive) for tp in types):
            if self.name is None:
                self.name = ""
                for prim in types:
                    self.name += prim.name
            self.types = types
        else:
            raise TypeError(f"Expecting a primitive type string or a list of Primitives, got : {types}")
        Prim_.__init__(self, self.types, self.name)


class Call(Call_):
    r"""
    Express a primitive CNode.
    """
    def __init__(self, prim_pattern, inputs=None):
        r"""
        Args:
            prim_pattern (Union[str, :class:`mindspore.graph_utils.graph_pattern.IsPrimTypeOf`,
                          :class:`mindspore.ops.Primitive`]): Primitive ValueNode in the Primitive CNode.
            inputs (Union[list[:class:`mindspore.graph_utils.graph_pattern`],
                          tuple[:class:`mindspore.graph_utils.graph_pattern`]]):
                Specify inputs pattern for the primitive(s), optional. If None, accepts any inputs; if specified, input
                patterns should be of right order and each element should be one of the exposed Pattern instance.

        Raises:
            TypeError: raise type error for invalid argument.
        """
        if not isinstance(prim_pattern, (Pattern, str, Primitive)):
            raise TypeError(f"Expect prim_pattern to be Pattern, Primitive or string,  got : {prim_pattern}")
        self.prim_pattern = prim_pattern
        self.inputs = []
        if inputs is None:
            pass
        elif isinstance(inputs, (tuple, list)) and all(isinstance(input, Pattern) for input in inputs):
            self.inputs = inputs
        else:
            raise TypeError(f"Expect inputs to be a list of Patterns, got : {inputs}")
        Call_.__init__(self, self.prim_pattern, self.inputs)


class NoneOf(NoneOf_):
    r"""
    Express a pattern which forbids a list of patterns.

    NOTE:
        NoneOf pattern should not be the root pattern.
    """
    def __init__(self, patterns=None):
        r"""
        Args:
            patterns(Union[list[:class:`mindspore.graph_utils.graph_pattern`]]: list of forbidden patterns, each
            element should be one of the exposed Pattern instance.

        Raises:
            TypeError: raise type error for invalid argument.
        """
        self.patterns = patterns
        if patterns is None:
            NoneOf_.__init__(self, ())
        elif isinstance(patterns, Pattern):
            NoneOf_.__init__(self, [patterns])
        elif isinstance(patterns, (tuple, list)) and all(isinstance(pattern, Pattern) for pattern in patterns):
            NoneOf_.__init__(self, patterns)
        else:
            raise TypeError(f"Expect list of Patterns/Pattern, got : {patterns}")


class NewTensor(NewTensor_):
    r"""
    New Tensor to be used in the target.
    """
    def __init__(self, input_tensor):
        r"""
        Args:
            input_tensor(:class:`mindspore.common.tensor.Tensor`): new tensor to be used in the target.

        Raises:
            TypeError: raise type error for invalid argument.
        """
        self.input_tensor = input_tensor
        if isinstance(input_tensor, Tensor):
            NewTensor_.__init__(self, input_tensor)
        else:
            raise TypeError(f"Expect input_tensor to be a Tensor， got : {input_tensor}")


class NewParameter(NewParameter_):
    r"""
    New Parameter to be used in the target.
    """
    def __init__(self, para_name, default_tensor, requires_grad=False, layerwise_parallel=False):
        r"""
        Args:
            para_name(str): name for the new Parameter.
            default_tensor(:class:`mindspore.common.tensor.Tensor`): default value for the new Parameter.
            requires_grad(bool): True if the parameter requires gradient. Default: True.
            layerwise_parallel(bool): switch for layerwise parallel mode. Default: False.

        Raises:
            TypeError: raise type error for invalid argument.
        """
        self.para_name = para_name
        self.default_tensor = default_tensor
        self.requires_grad = requires_grad
        self.layerwise_parallel = layerwise_parallel
        if isinstance(para_name, str) and isinstance(default_tensor, Tensor) and isinstance(requires_grad, bool) and\
           isinstance(layerwise_parallel, bool):
            NewParameter_.__init__(self, self.para_name, self.default_tensor, self.requires_grad,
                                   self.layerwise_parallel)
        else:
            raise TypeError(f"Expect para_name(str), default_tensor(Tensor), requires_grad(bool), \
                              layerwise_parallel(bool)， got : {para_name}, {default_tensor}, \
                              {requires_grad}, {layerwise_parallel}")

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
from mindspore._c_expression import Pattern, IsIn_, IsPrimTypeOf_, CallWith_, IsNot_, AnyPattern, NewTensor_

__all__ = [
    "IsIn",
    "IsPrimTypeOf",
    "CallWith",
    "IsNot",
    "AnyPattern",
    "NewTensor",
]

class IsIn(IsIn_):
    """
    Express a pattern which allows a list of patterns.
    """
    def __init__(self, patterns=None, should_replace=True):
        r"""
        Args:
            patterns(list/tuple): list of allowed patterns
            should_replace(bool): added this for interface consistency. Should only set this in sub-patterns.
        """
        if not should_replace:
            raise ValueError("IsIn pattern does not have its own should_replace attribute. Set should_replace in \
                              its sub-pattern instead.")
        self.patterns = patterns
        if patterns is None:
            IsIn_.__init__(self, ())
        elif isinstance(patterns, Pattern):
            IsIn_.__init__(self, [patterns])
        elif isinstance(patterns, (tuple, list)) and all(isinstance(pattern, Pattern) for pattern in patterns):
            IsIn_.__init__(self, patterns)
        else:
            raise TypeError(f"Expect patterns to be a list of Patterns/Pattern, got : {patterns}")

class IsPrimTypeOf(IsPrimTypeOf_):
    r"""
    Express a pattern of certain primitive type(s).
    NOTE: This pattern will match and only match the primitive value node. If matching primitive CNode is needed,
    please refer to CallWith pattern.
    """
    def __init__(self, types, name=None, should_replace=True):
        r"""
        Args:
            types (str/(list/tuple of Primitives)): Specify allowed types.
                If it is a string, the form could be
                    1) a single primitive type, e.g. 'Conv2D'
                    2) a set of primitive types separated by '|', e.g. 'MatMul|Conv2D'
                It can also be a list of Primitives, e.g. [ops.Conv2D(1, 6)]
            name (str): name of the pattern, optional
            should_replace
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
        IsPrimTypeOf_.__init__(self, self.types, self.name, should_replace)

class CallWith(CallWith_):
    r"""
    Express a primitive CNode.
    """
    def __init__(self, prim_pattern, inputs=None, should_replace=False):
        r"""
        Args:
            prim_pattern (Pattern/Primitive/str): Primitive ValueNode in the Primitive CNode.
            inputs (list/tuple): Specify inputs pattern for the primitive(s), optional. If None, accepts any inputs;
                if specified, input patterns should be of right order.
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
        CallWith_.__init__(self, self.prim_pattern, self.inputs, should_replace)


class IsNot(IsNot_):
    r"""
    Express a pattern which forbids a list of patterns.
    NOTE: IsNot pattern should not be the root pattern.
    """
    def __init__(self, patterns=None, should_replace=True):
        r"""
        Args:
            patterns(list/tuple): list of forbiden patterns
            should_replace(bool): added this for interface consistency. Should only set this in sub-patterns.
        """
        if not should_replace:
            raise ValueError("IsNot pattern does not have its own should_replace attribute. Set should_replace in \
                              its sub-pattern instead.")
        self.patterns = patterns
        if patterns is None:
            IsNot_.__init__(self, ())
        elif isinstance(patterns, Pattern):
            IsNot_.__init__(self, [patterns])
        elif isinstance(patterns, (tuple, list)) and all(isinstance(pattern, Pattern) for pattern in patterns):
            IsNot_.__init__(self, patterns)
        else:
            raise TypeError(f"Expect list of Patterns/Pattern, got : {patterns}")

class NewTensor(NewTensor_):
    r"""
    New Tensor to be used in the target.
    """
    def __init__(self, input_tensor, should_replace=False):
        r"""
        Args:
            input_tensor(Tensor): new tensor to be used in the target
            should_replace(bool): added this for interface consistency. NewTensor should only appear in the target.
        """
        if should_replace:
            raise ValueError("NewTensor should only appear in the target, thus should_replace can onlyu be False.")
        self.input_tensor = input_tensor
        if isinstance(input_tensor, Tensor):
            NewTensor_.__init__(self, input_tensor)
        else:
            raise TypeError(f"Expect input_tensor to be a Tensor， got : {input_tensor}")

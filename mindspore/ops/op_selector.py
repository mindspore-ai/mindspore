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

"""
A factory class that create op selector instance to config switch on a class,
which can be used to control the switch of op type: GraphKernel or Primitive.
"""
import importlib
import inspect
from mindspore import context


class _OpSelector:
    """
    A helper class, which can be used to choose different type of operator.

    When an instance of this class is called, we return the right operator
    according to the context['enable_graph_kernel'] and the name of the
    parameter. returned operator will be a GraphKernel op ora  Primitive op.

    Args:
        op (class): an empty class has an operator name as its class name
        config_optype (str): operator type, which must be either 'GraphKernel'
        or 'Primitive'
        graph_kernel_pkg (str): real operator's package name
        primitive_pkg (str): graph kernel operator's package name

    Examples:
        >>> class A: pass
        >>> selected_op = _OpSelector(A, "GraphKernel",
        ...                           "graph_kernel.ops.pkg", "primitive.ops.pkg")
        >>> # selected_op() will call graph_kernel.ops.pkg.A()
    """
    GRAPH_KERNEL = "GraphKernel"
    PRIMITIVE = "Primitive"
    DEFAULT_OP_TYPE = PRIMITIVE
    KW_STR = "op_type"

    def __init__(self, op, config_optype, primitive_pkg, graph_kernel_pkg):
        self.op_name = op.__name__
        self.config_optype = config_optype
        self.graph_kernel_pkg = graph_kernel_pkg
        self.primitive_pkg = primitive_pkg

    def __call__(self, *args, **kwargs):
        _op_type = _OpSelector.DEFAULT_OP_TYPE
        if context.get_context("device_target") in ['Ascend', 'GPU'] and context.get_context("enable_graph_kernel"):
            if _OpSelector.KW_STR in kwargs:
                _op_type = kwargs.get(_OpSelector.KW_STR)
                kwargs.pop(_OpSelector.KW_STR, None)
            elif self.config_optype is not None:
                _op_type = self.config_optype
        if _op_type == _OpSelector.GRAPH_KERNEL:
            pkg = self.graph_kernel_pkg
        else:
            pkg = self.primitive_pkg
        op = getattr(importlib.import_module(pkg, __package__), self.op_name)
        return op(*args, **kwargs)


def new_ops_selector(primitive_pkg, graph_kernel_pkg):
    """
    A factory method to return an op selector

    When the GraphKernel switch is on:
        `context.get_context('enable_graph_kernel') == True`, we have 2 ways to control the op type:
        (1). call the real op with an extra parameter `op_type='Primitive'` or `op_type='GraphKernel'`
        (2). pass a parameter to the op selector, like `@op_selector('Primitive')` or
                `@op_selector('GraphKernel')`
        (3). default op type is PRIMITIVE
        The order of the highest priority to lowest priority is (1), (2), (3)
    If the GraphKernel switch is off, then op_type will always be PRIMITIVE.

    Args:
        primitive_pkg (str): primitive op's package name
        graph_kernel_pkg (str): graph kernel op's package name

    Returns:
        returns an op selector, which can control what operator should be actually called.

    Examples:
        >>> op_selector = new_ops_selector("primitive_pkg.some.path",
        ...                                "graph_kernel_pkg.some.path")
        >>> @op_selector
        >>> class ReduceSum: pass
    """

    def op_selector(cls_or_optype):

        _primitive_pkg = primitive_pkg
        _graph_kernel_pkg = graph_kernel_pkg

        def direct_op_type():
            darg = None
            if cls_or_optype is None:
                pass
            elif not inspect.isclass(cls_or_optype):
                darg = cls_or_optype
            return darg

        if direct_op_type() is not None:
            def deco_cls(_real_cls):
                return _OpSelector(_real_cls, direct_op_type(), _primitive_pkg, _graph_kernel_pkg)
            return deco_cls

        return _OpSelector(cls_or_optype, direct_op_type(), _primitive_pkg, _graph_kernel_pkg)

    return op_selector

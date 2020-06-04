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
which can be used to control the switch of op type: black or white block.
"""
import importlib
import inspect
from mindspore import context


class OpSelector:
    """
    Works as a proxy.
    When CALLED on an instance of this class, resolve the parameter to the
    corresponding white block op or black block op.

    example:
    op_selector = OpSelector.new_ops_selector("black.ops.pkg", "white.ops.pkg")

    @op_selector
    class ReduceMean: pass
    """
    GRAPH_KERNEL = "GraphKernel"
    PRIMITIVE = "Primitive"
    DEFAULT_OP_TYPE = PRIMITIVE
    KW_STR = "op_type"

    def __init__(self, op, config_optype, black_block_pkg, white_block_pkg):
        self.op_name = op.__name__
        self.config_optype = config_optype
        self.white_block_pkg = white_block_pkg
        self.black_block_pkg = black_block_pkg

    def __call__(self, *args, **kwargs):
        _op_type = OpSelector.DEFAULT_OP_TYPE
        if context.get_context("enable_graph_kernel"):
            if OpSelector.KW_STR in kwargs:
                _op_type = kwargs.get(OpSelector.KW_STR)
                kwargs.pop(OpSelector.KW_STR, None)
            elif self.config_optype is not None:
                _op_type = self.config_optype
        if _op_type == OpSelector.GRAPH_KERNEL:
            pkg = self.white_block_pkg
        else:
            pkg = self.black_block_pkg
        op = getattr(importlib.import_module(pkg, __package__), self.op_name)
        return op(*args, **kwargs)

    @staticmethod
    def new_ops_selector(black_block_pkg, white_block_pkg):
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
        """
        def op_selector(cls_or_optype):

            _black_block_pkg = black_block_pkg
            _white_block_pkg = white_block_pkg

            def direct_op_type():
                darg = None
                if cls_or_optype is None:
                    pass
                elif not inspect.isclass(cls_or_optype):
                    darg = cls_or_optype
                return darg

            if direct_op_type() is not None:
                def deco_cls(_real_cls):
                    return OpSelector(_real_cls, direct_op_type(), _black_block_pkg, _white_block_pkg)
                return deco_cls

            return OpSelector(cls_or_optype, direct_op_type(), _black_block_pkg, _white_block_pkg)

        return op_selector

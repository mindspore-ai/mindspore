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
"""Gain bnn_with_loss by rewrite WithLossCell as WithBNNLossCell to suit for BNN model"""
import ast
import importlib
import os
import sys
import tempfile
import astunparse
import mindspore


class _CodeTransformer(ast.NodeTransformer):
    """
    Add kl_loss computation by analyzing the python code structure with the help of the AST module.

    Args:
        layer_count (int): The number of kl loss to be generated, namely the number of Bayesian layers.
    """

    def __init__(self, layer_count):
        self.layer_count = layer_count

    def visit_FunctionDef(self, node):
        """visit function and add kl_loss computation."""
        self.generic_visit(node)
        if node.name == 'cal_kl_loss':
            for i in range(self.layer_count):
                func = ast.Assign(targets=[ast.Name(id='loss', ctx=ast.Store())],
                                  value=ast.BinOp(left=ast.Name(id='loss', ctx=ast.Load()), op=ast.Add(),
                                                  right=ast.Call(func=ast.Name(id='self.kl_loss' + '[' + str(i) + ']',
                                                                               ctx=ast.Load()),
                                                                 args=[], keywords=[])))
                node.body.insert(-1, func)
        return node


def _generate_kl_loss_func(layer_count):
    """Rewrite WithLossCell as WithBNNLossCell to suit for BNN model."""
    path = os.path.dirname(mindspore.__file__) + '/nn/probability/transforms/bnn_loss/withLossCell.py'
    with open(path, 'r') as fp:
        srclines = fp.readlines()
    src = ''.join(srclines)
    if src.startswith((' ', '\t')):
        src = 'if 1:\n' + src
    expr_ast = ast.parse(src, mode='exec')
    transformer = _CodeTransformer(layer_count)
    modify = transformer.visit(expr_ast)
    modify = ast.fix_missing_locations(modify)
    func = astunparse.unparse(modify)
    return func


def gain_bnn_with_loss(layer_count, backbone, loss_fn, dnn_factor, bnn_factor):
    """
    Gain bnn_with_loss, which wraps bnn network with loss function and kl loss of each bayesian layer.

    Args:
        layer_count (int): The number of kl loss to be generated, namely the number of Bayesian layers.
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
        dnn_factor (int, float): The coefficient of backbone's loss, which is computed by loss function.
        bnn_factor (int, float): The coefficient of kl loss, which is kl divergence of Bayesian layer.
    """
    bnn_loss_func = _generate_kl_loss_func(layer_count)
    path = os.path.dirname(mindspore.__file__)
    bnn_loss_file = tempfile.NamedTemporaryFile(mode='w+t', suffix='.py', delete=True,
                                                dir=path + '/nn/probability/transforms/bnn_loss')
    bnn_loss_file.write(bnn_loss_func)
    bnn_loss_file.seek(0)

    sys.path.append(path + '/nn/probability/transforms/bnn_loss')

    module_name = os.path.basename(bnn_loss_file.name)[0:-3]
    bnn_loss_module = importlib.import_module(module_name, __package__)
    bnn_with_loss = bnn_loss_module.WithBNNLossCell(backbone, loss_fn, dnn_factor, bnn_factor)
    return bnn_with_loss, bnn_loss_file

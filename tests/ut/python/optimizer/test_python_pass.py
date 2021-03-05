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
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import _constants as Constants
from mindspore.graph_utils.python_pass import registe_pass, unregiste_pass, set_renorm, gen_new_parameter,\
    cancel_new_parameter, set_reopt
from mindspore.common.api import _generate_pip_args
from mindspore._c_expression import generate_key, Executor_
from mindspore.graph_utils.graph_pattern import OneOf, Prim, Call, NoneOf, Any, NewTensor, NewParameter, Imm

context.set_context(mode=context.GRAPH_MODE)

def get_func_graph(obj, *args, phase="validate"):
    args_names, args_list = _generate_pip_args(obj, *args)
    dic = dict(zip(args_names, args_list))
    key = generate_key(phase, dic)
    phase_prefix = str(key[1])
    if phase == 'export':
        phase = phase + '.' + phase_prefix + '.' + str(obj.create_time)
    else:
        phase = phase_prefix + phase + '.' + str(obj.create_time)
    _executor = Executor_.get_instance()
    _executor.compile(obj, args_list, phase, False)
    return _executor.get_func_graph(phase)

def test_softmax_relu():
    """
    Use python pass to transform from Softmax to ReLU.
    """
    inputs = Tensor(np.ones([42]), mindspore.float16)
    softmax_model = nn.Softmax()

    @registe_pass(run_only_once=True)
    def softmax_relu_pass():
        x = Any()
        pattern = Call(P.Softmax(), [x])
        target = Call(P.ReLU(), [x])
        return pattern, target

    transformed_repr = get_func_graph(softmax_model, inputs).get_return().expanded_str(2)
    unregiste_pass(softmax_relu_pass)
    assert "ReLU" in transformed_repr
    assert "Softmax" not in transformed_repr

def test_prim():
    inputs = Tensor(np.ones([42]), mindspore.float16)
    softmax_model = nn.Softmax()

    @registe_pass(run_only_once=True)
    def softmax_relu_pass():
        x = Any()
        sigmoid_softmax_pattern = Prim([P.Sigmoid(), P.Softmax()])
        pattern = Call(sigmoid_softmax_pattern, [x])
        target = Call(P.ReLU(), [x])
        return pattern, target

    transformed_repr = get_func_graph(softmax_model, inputs).get_return().expanded_str(3)
    unregiste_pass(softmax_relu_pass)
    assert "ReLU" in transformed_repr
    assert "Softmax" not in transformed_repr

def test_softmax_relu_sigmoid():
    """
    Use python pass to transform from Softmax(x) to ReLU(Sigmoid(x)).

    NOTE:
        Sigmoid pattern only exists in the target.
    """
    inputs = Tensor(np.ones([42]), mindspore.float16)
    softmax_model = nn.Softmax()

    @registe_pass(run_only_once=True)
    def softmax_relu_pass():
        x = Any()
        softmax_pattern = Prim(P.Softmax())
        pattern = Call(softmax_pattern, [x])
        sigmoid_pattern = Prim(P.Sigmoid())
        call_sigmoid = Call(sigmoid_pattern, [x])
        relu_pattern = Prim(P.ReLU())
        target = Call(relu_pattern, [call_sigmoid])
        return pattern, target

    transformed_repr = get_func_graph(softmax_model, inputs).get_return().expanded_str(3)
    unregiste_pass(softmax_relu_pass)
    assert "ReLU" in transformed_repr
    assert "Sigmoid" in transformed_repr
    assert "Softmax" not in transformed_repr


def test_isin_pattern_0():
    """
    Test IsIn pattern which expresses the IsIn/OneOf semantics.
    """
    inputs = Tensor(np.ones([42]), mindspore.float16)
    softmax_model = nn.Softmax()

    @registe_pass(run_only_once=True)
    def softmax_relu_pass():
        x = Any()
        softmax_pattern = Prim(P.Softmax())
        call_softmax = Call(softmax_pattern, [x])
        relu_pattern = Prim(P.ReLU())
        call_relu = Call(relu_pattern, [x])

        pattern = OneOf([call_softmax, call_relu])
        relu6_pattern = Prim(P.ReLU6())
        target = Call(relu6_pattern, [x])
        return pattern, target
    transformed_repr = get_func_graph(softmax_model, inputs).get_return().expanded_str(2)
    unregiste_pass(softmax_relu_pass)
    assert "ReLU6" in transformed_repr
    assert "Softmax" not in transformed_repr

def test_isin_pattern_1():
    """
    Test IsIn. IsIn is used as nested inputs for the target in this case.
    """
    inputs = Tensor(np.ones([42]), mindspore.float16)
    softmax_model = nn.Softmax()

    @registe_pass(run_only_once=True)
    def softmax_neg_pass():
        x = Any()
        softmax_pattern = Prim(P.Softmax())
        call_softmax = Call(softmax_pattern, [x])
        relu_pattern = Prim(P.ReLU())
        call_relu = Call(relu_pattern, [x])

        pattern = OneOf([call_softmax, call_relu])
        neg_ops = Prim(P.Neg())
        target = Call(neg_ops, [pattern])
        return pattern, target
    transformed_repr = get_func_graph(softmax_model, inputs).get_return().expanded_str(4)
    unregiste_pass(softmax_neg_pass)
    assert "Neg" in transformed_repr
    assert "Softmax" in transformed_repr

def test_isnot_pattern_0():
    """
    Test IsNot pattern which expresses the IsNot semantics.
    Case: IsNot pass failed to match
    """
    set_renorm(False)
    set_reopt(False)
    class ConvBN(nn.Cell):
        def __init__(self):
            super(ConvBN, self).__init__()
            self.conv = P.Conv2D(32, 3)
            self.conv_weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
            self.scale = Tensor(np.ones([32]), mindspore.float32)
            self.bias = Tensor(np.ones([32]), mindspore.float32)
            self.mean = Tensor(np.ones([32]), mindspore.float32)
            self.variance = Tensor(np.ones([32]), mindspore.float32)
            self.bn = P.BatchNorm()
        def construct(self, x):
            x = self.conv(x, self.conv_weight)
            x = self.bn(x, self.scale, self.bias, self.mean, self.variance)
            return x
    inputs = Tensor(np.random.normal(0, 1, (10, 32, 32, 32)), mindspore.float32)
    conv_bn_model = ConvBN()

    @registe_pass(requires_grad=False, run_only_once=True)
    def single_bn_pass():
        """
        Sub a BN which does NOT take Conv as inputs to ReLU6.
        """
        conv2d_prim = Prim("Conv2D")
        conv2d = Call(conv2d_prim)
        pattern_0 = NoneOf(conv2d)
        pattern = Call(P.BatchNorm(), [pattern_0])
        target = Call(P.ReLU6(), [pattern_0])
        return pattern, target

    @registe_pass(requires_grad=False, run_only_once=True)
    def bn_pass():
        """
        Sub a BN to Softmax.
        """
        pattern = Call(P.BatchNorm())
        target = Call(P.Softmax())
        return pattern, target

    transformed_repr = get_func_graph(conv_bn_model, inputs).get_return().expanded_str(5)
    unregiste_pass(single_bn_pass)
    unregiste_pass(bn_pass)
    assert "ReLU6" not in transformed_repr
    assert "Softmax" in transformed_repr
    set_renorm(True)

def test_isnot_pattern_1():
    """
    Test IsNot pattern which expresses the IsNot semantics.
    Case: IsNot pattern matches with the graph
    """
    inputs = Tensor(np.ones([42]), mindspore.float16)
    softmax_model = nn.Softmax()

    @registe_pass(run_only_once=True)
    def single_bn_pass():
        """
        Sub a BN which does NOT take MatMul as inputs to ReLU6.
        """
        matmul = Prim("MatMul")
        pattern_0 = NoneOf(matmul)
        softmax = P.Softmax()
        pattern = Call(softmax, [pattern_0])
        relu6 = P.ReLU6()
        target = Call(relu6, [pattern_0])
        return pattern, target

    transformed_repr = get_func_graph(softmax_model, inputs).get_return().expanded_str(5)
    unregiste_pass(single_bn_pass)
    assert "ReLU6" in transformed_repr
    assert "Softmax" not in transformed_repr

def test_newtensor_pattern():
    """
    Test NewTensor pattern in the target
    """
    set_renorm(False)
    set_reopt(False)
    inputs = Tensor(np.ones([42]), mindspore.float16)
    softmax_model = nn.Softmax()

    @registe_pass(requires_grad=False, run_only_once=True)
    def softmax_addn_pass():
        x = Any()
        pattern = Call(P.Softmax(), [x])

        weight_tensor = Tensor(np.zeros([42]), mindspore.float16)
        new_weight = NewTensor(weight_tensor)
        target = Call(P.AddN(), [x, new_weight])
        return pattern, target
    transformed_repr = get_func_graph(softmax_model, inputs).get_return().expanded_str(2)
    unregiste_pass(softmax_addn_pass)
    assert "AddN" in transformed_repr
    assert "Softmax" not in transformed_repr
    set_renorm(True)

def test_newparameter_pattern():
    """
    Test NewParameter pattern in the target
    """
    inputs = Tensor(np.ones([42]), mindspore.float16)
    softmax_model = nn.Softmax()

    set_renorm(False)
    set_reopt(False)
    @registe_pass(requires_grad=False, run_only_once=True)
    def softmax_addn_pass():
        x = Any()
        pattern = Call(P.Softmax(), [x])

        default_tensor0 = Tensor(np.ones((4, 4)), mindspore.float32)
        default_tensor1 = Tensor(np.ones((4, 4)), mindspore.float32)
        new_para_0 = NewParameter("Merlin", default_tensor0)
        new_para_1 = NewParameter("Arthur", default_tensor1)
        target_0 = Call(P.MatMul(), [new_para_0, new_para_1])
        target = Call("MakeTuple", [target_0])
        return pattern, target
    transformed_repr = get_func_graph(softmax_model, inputs).get_return().expanded_str(5)
    unregiste_pass(softmax_addn_pass)
    assert "MatMul" in transformed_repr
    assert "MakeTuple" in transformed_repr
    assert "Softmax" not in transformed_repr

def test_imm_target():
    """
    Test NewParameter pattern in the target
    """
    inputs = Tensor(np.ones([42]), mindspore.float16)
    softmax_model = nn.Softmax()

    set_renorm(False)
    set_reopt(False)
    @registe_pass(requires_grad=False, run_only_once=True)
    def softmax_pass():
        x = Any()
        pattern = Call(P.Softmax(), [x])
        imm = Imm(0)
        target_0 = Call("MakeTuple", [pattern])
        target = Call(Constants.kTupleGetItem, [target_0, imm])
        return pattern, target
    transformed_repr = get_func_graph(softmax_model, inputs).get_return().expanded_str(5)
    unregiste_pass(softmax_pass)
    assert "MakeTuple" in transformed_repr
    assert Constants.kTupleGetItem in transformed_repr
    assert "Softmax" in transformed_repr

def test_gen_new_parameter():
    """
    Test gen_new_parameter
    """
    inputs = Tensor(np.ones([42]), mindspore.float16)
    softmax_model = nn.Softmax()

    default_tensor = Tensor(np.ones((4, 4)), mindspore.float32)
    new_para = NewParameter("Merlin", default_tensor)
    set_renorm(False)
    set_reopt(False)
    gen_new_parameter(new_para)
    @registe_pass(requires_grad=False, run_only_once=True)
    def softmax_make_tuple_pass():
        x = Any()
        softmax = P.Softmax()
        pattern = Call(softmax, [x])

        target = Call("MakeTuple", [pattern, new_para])
        return pattern, target
    transformed_repr = get_func_graph(softmax_model, inputs).get_return().expanded_str(5)
    assert "Merlin" in transformed_repr
    unregiste_pass(softmax_make_tuple_pass)
    cancel_new_parameter(new_para)
    transformed_repr = get_func_graph(softmax_model, inputs).get_return().expanded_str(5)
    assert "Merlin" not in transformed_repr

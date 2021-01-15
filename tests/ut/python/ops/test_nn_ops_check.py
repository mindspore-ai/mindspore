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
""" test ops """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config_exception


class Conv2DBackpropInputNet(nn.Cell):
    def __init__(self, net, x_shape):
        super(Conv2DBackpropInputNet, self).__init__()
        self.net = net
        self.x_shape = x_shape

    def construct(self, dout, w):
        return self.net(dout, w, self.x_shape)


class TopKNet(nn.Cell):
    def __init__(self, net, k):
        super(TopKNet, self).__init__()
        self.net = net
        self.k = k

    def construct(self, x):
        return self.net(x, self.k)


raise_set = [
    # input is scalar
    ('Flatten0', {
        'block': (P.Flatten(), {'exception': TypeError, 'error_keywords': ['Flatten']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # dim of input is zero
    ('Flatten1', {
        'block': (P.Flatten(), {'exception': ValueError, 'error_keywords': ['Flatten']}),
        'desc_inputs': [F.scalar_to_tensor(5.0)],
        'skip': ['backward']}),

    # input is scalar
    ('Softmax0', {
        'block': (P.Softmax(), {'exception': TypeError, 'error_keywords': ['Softmax']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # axis is empty tuple
    ('Softmax1', {
        'block': (P.Softmax(axis=()), {'exception': ValueError, 'error_keywords': ['Softmax']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    # axis value is not in range
    ('Softmax2', {
        'block': (P.Softmax(axis=2), {'exception': ValueError, 'error_keywords': ['Softmax']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),

    # input is scalar
    ('LogSoftmax0', {
        'block': (P.LogSoftmax(), {'exception': TypeError, 'error_keywords': ['LogSoftmax']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # axis value is not in range
    ('LogSoftmax1', {
        'block': (P.LogSoftmax(axis=2), {'exception': ValueError, 'error_keywords': ['LogSoftmax']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.float32))],
        'skip': ['backward']}),

    # input is scalar
    ('ReLU0', {
        'block': (P.ReLU(), {'exception': TypeError, 'error_keywords': ['ReLU']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(Bool)
    ('ReLU1', {
        'block': (P.ReLU(), {'exception': TypeError, 'error_keywords': ['ReLU']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.bool_))],
        'skip': ['backward']}),

    # input is scalar
    ('ReLU60', {
        'block': (P.ReLU6(), {'exception': TypeError, 'error_keywords': ['ReLU6']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(int32)
    ('ReLU61', {
        'block': (P.ReLU6(), {'exception': TypeError, 'error_keywords': ['ReLU6']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32))],
        'skip': ['backward']}),

    # input is scalar
    ('Elu0', {
        'block': (P.Elu(), {'exception': TypeError, 'error_keywords': ['Elu']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(int32)
    ('Elu1', {
        'block': (P.Elu(), {'exception': TypeError, 'error_keywords': ['Elu']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32))],
        'skip': ['backward']}),

    # input is scalar
    ('Sigmoid0', {
        'block': (P.Sigmoid(), {'exception': TypeError, 'error_keywords': ['Sigmoid']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(int32)
    ('Sigmoid1', {
        'block': (P.Sigmoid(), {'exception': TypeError, 'error_keywords': ['Sigmoid']}),
        'desc_inputs': [Tensor(np.ones([3, 4]).astype(np.int32))],
        'skip': ['backward']}),

    # input is scalar
    ('Tanh0', {
        'block': (P.Tanh(), {'exception': TypeError, 'error_keywords': ['Tanh']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),

    # input is scalar
    ('BatchNorm0', {
        'block': (P.BatchNorm(is_training=False), {'exception': TypeError, 'error_keywords': ['BatchNorm']}),
        'desc_inputs': [5.0, 5.0, 5.0, 5.0, 5.0],
        'skip': ['backward']}),
    # is_training=False and mean=None
    ('BatchNorm1', {
        'block': (P.BatchNorm(is_training=False), {'exception': TypeError, 'error_keywords': ['BatchNorm']}),
        'desc_inputs': [Tensor(np.ones([5, 3]).astype(np.float32)), Tensor(np.ones([5, 3]).astype(np.float32)),
                        Tensor(np.ones([5, 3]).astype(np.float32)), None, None],
        'skip': ['backward']}),
    # is_training=True and mean=None
    ('BatchNorm2', {
        'block': (P.BatchNorm(is_training=True), {'exception': TypeError, 'error_keywords': ['BatchNorm']}),
        'desc_inputs': [Tensor(np.ones([5, 3]).astype(np.float32)), Tensor(np.ones([3]).astype(np.float32)),
                        Tensor(np.ones([3]).astype(np.float32)), Tensor(np.ones([3]).astype(np.float16)),
                        Tensor(np.ones([3]).astype(np.float32))],
        'skip': ['backward']}),
    # scale and bias rank > 1
    ('BatchNorm3', {
        'block': (P.BatchNorm(is_training=True), {'exception': ValueError, 'error_keywords': ['BatchNorm']}),
        'desc_inputs': [Tensor(np.ones([5, 3]).astype(np.float32)), Tensor(np.ones([5, 3]).astype(np.float32)),
                        Tensor(np.ones([5, 3]).astype(np.float32)), Tensor(np.ones([3]).astype(np.float32)),
                        Tensor(np.ones([3]).astype(np.float32))],
        'skip': ['backward']}),
    # scale and bias shape not match
    ('BatchNorm4', {
        'block': (P.BatchNorm(is_training=True), {'exception': ValueError, 'error_keywords': ['BatchNorm']}),
        'desc_inputs': [Tensor(np.ones([5, 3]).astype(np.float32)), Tensor(np.ones([3]).astype(np.float32)),
                        Tensor(np.ones([7]).astype(np.float32)), Tensor(np.ones([3]).astype(np.float32)),
                        Tensor(np.ones([3]).astype(np.float32))],
        'skip': ['backward']}),
    # is_training=False, mean and variance shape not match
    ('BatchNorm5', {
        'block': (P.BatchNorm(is_training=False), {'exception': ValueError, 'error_keywords': ['BatchNorm']}),
        'desc_inputs': [Tensor(np.ones([5, 3]).astype(np.float32)), Tensor(np.ones([3]).astype(np.float32)),
                        Tensor(np.ones([3]).astype(np.float32)), Tensor(np.ones([3]).astype(np.float32)),
                        Tensor(np.ones([5]).astype(np.float32))],
        'skip': ['backward']}),
    # is_training=False, mean and scale shape not match
    ('BatchNorm6', {
        'block': (P.BatchNorm(is_training=False), {'exception': ValueError, 'error_keywords': ['BatchNorm']}),
        'desc_inputs': [Tensor(np.ones([5, 3]).astype(np.float32)), Tensor(np.ones([3]).astype(np.float32)),
                        Tensor(np.ones([3]).astype(np.float32)), Tensor(np.ones([5]).astype(np.float32)),
                        Tensor(np.ones([5]).astype(np.float32))],
        'skip': ['backward']}),

    # input is scalar
    ('Conv2D0', {
        'block': (P.Conv2D(2, (5, 5)), {'exception': TypeError, 'error_keywords': ['Conv2D']}),
        'desc_inputs': [5.0, 5.0],
        'skip': ['backward']}),
    # input is Tensor(bool)
    ('Conv2D1', {
        'block': (P.Conv2D(2, (5, 5)), {'exception': TypeError, 'error_keywords': ['Conv2D']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.bool_)), Tensor(np.ones([5]).astype(np.bool_))],
        'skip': ['backward']}),
    # input x and w type mismatch
    ('Conv2D2', {
        'block': (P.Conv2D(2, (5, 5)), {'exception': TypeError, 'error_keywords': ['Conv2D']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.float32)), Tensor(np.ones([5]).astype(np.float16))],
        'skip': ['backward']}),
    # rank of x is not 4
    ('Conv2D3', {
        'block': (P.Conv2D(2, (5, 5)), {'exception': ValueError, 'error_keywords': ['Conv2D']}),
        'desc_inputs': [Tensor(np.ones([1, 1]).astype(np.float32)), Tensor(np.ones([1, 1, 9, 9]).astype(np.float32))],
        'skip': ['backward']}),
    # rank of 2 is not 4
    ('Conv2D4', {
        'block': (P.Conv2D(2, (5, 5)), {'exception': ValueError, 'error_keywords': ['Conv2D']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 9, 9]).astype(np.float32)),
                        Tensor(np.ones([1, 1, 9]).astype(np.float32))],
        'skip': ['backward']}),
    # x_shape[1] / group != w_shape[1]
    ('Conv2D5', {
        'block': (P.Conv2D(2, (5, 5)), {'exception': ValueError, 'error_keywords': ['Conv2D']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 9, 9]).astype(np.float32)),
                        Tensor(np.ones([1, 2, 9, 9]).astype(np.float32))],
        'skip': ['backward']}),
    # out_channel != w_shape[0]
    ('Conv2D6', {
        'block': (P.Conv2D(2, (5, 5)), {'exception': ValueError, 'error_keywords': ['Conv2D']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 9, 9]).astype(np.float32)),
                        Tensor(np.ones([1, 1, 9, 9]).astype(np.float32))],
        'skip': ['backward']}),
    # kernel_size != w_shape[2:4]
    ('Conv2D7', {
        'block': (P.Conv2D(2, (5, 5)), {'exception': ValueError, 'error_keywords': ['Conv2D']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 9, 9]).astype(np.float32)),
                        Tensor(np.ones([2, 1, 5, 6]).astype(np.float32))],
        'skip': ['backward']}),

    # input is scalar
    ('DepthwiseConv2dNative0', {
        'block': (P.DepthwiseConv2dNative(2, (5, 5)),
                  {'exception': TypeError, 'error_keywords': ['DepthwiseConv2dNative']}),
        'desc_inputs': [5.0, 5.0],
        'skip': ['backward']}),
    # input is Tensor(bool)
    ('DepthwiseConv2dNative1', {
        'block': (P.DepthwiseConv2dNative(2, (5, 5)),
                  {'exception': TypeError, 'error_keywords': ['DepthwiseConv2dNative']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.bool_)), Tensor(np.ones([5]).astype(np.bool_))],
        'skip': ['backward']}),
    # input x and w type mismatch
    ('DepthwiseConv2dNative2', {
        'block': (P.DepthwiseConv2dNative(2, (5, 5)),
                  {'exception': TypeError, 'error_keywords': ['DepthwiseConv2dNative']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.float32)), Tensor(np.ones([5]).astype(np.float16))],
        'skip': ['backward']}),
    # rank of x is not 4
    ('DepthwiseConv2dNative3', {
        'block': (P.DepthwiseConv2dNative(2, (5, 5)),
                  {'exception': ValueError, 'error_keywords': ['DepthwiseConv2dNative']}),
        'desc_inputs': [Tensor(np.ones([1, 1]).astype(np.float32)), Tensor(np.ones([1, 1, 9, 9]).astype(np.float32))],
        'skip': ['backward']}),
    # rank of 2 is not 4
    ('DepthwiseConv2dNative4', {
        'block': (P.DepthwiseConv2dNative(2, (5, 5)),
                  {'exception': ValueError, 'error_keywords': ['DepthwiseConv2dNative']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 9, 9]).astype(np.float32)),
                        Tensor(np.ones([1, 1, 9]).astype(np.float32))],
        'skip': ['backward']}),
    # x_shape[1] != w_shape[1]
    ('DepthwiseConv2dNative5', {
        'block': (P.DepthwiseConv2dNative(2, (5, 5)),
                  {'exception': ValueError, 'error_keywords': ['DepthwiseConv2dNative']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 9, 9]).astype(np.float32)),
                        Tensor(np.ones([1, 2, 9, 9]).astype(np.float32))],
        'skip': ['backward']}),
    # kernel_size != w_shape[2:4]
    ('DepthwiseConv2dNative6', {
        'block': (P.DepthwiseConv2dNative(2, (5, 5)),
                  {'exception': ValueError, 'error_keywords': ['DepthwiseConv2dNative']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 9, 9]).astype(np.float32)),
                        Tensor(np.ones([2, 1, 5, 6]).astype(np.float32))],
        'skip': ['backward']}),

    # input is scalar
    ('MaxPoolWithArgmax0', {
        'block': (P.MaxPoolWithArgmax(), {'exception': TypeError, 'error_keywords': ['MaxPoolWithArgmax']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input is Tensor(bool)
    ('MaxPoolWithArgmax1', {
        'block': (P.MaxPoolWithArgmax(), {'exception': TypeError, 'error_keywords': ['MaxPoolWithArgmax']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.bool_))],
        'skip': ['backward']}),
    # rank of x is not 4
    ('MaxPoolWithArgmax2', {
        'block': (P.MaxPoolWithArgmax(), {'exception': ValueError, 'error_keywords': ['MaxPoolWithArgmax']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 32]).astype(np.float32))],
        'skip': ['backward']}),
    # kernel size is invalid(very large)
    ('MaxPoolWithArgmax3', {
        'block': (P.MaxPoolWithArgmax(kernel_size=50),
                  {'exception': ValueError, 'error_keywords': ['MaxPoolWithArgmax']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))],
        'skip': ['backward']}),

    # input is scalar
    ('MaxPool0', {
        'block': (P.MaxPool(), {'exception': TypeError, 'error_keywords': ['MaxPool']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # rank of x is not 4
    ('MaxPool1', {
        'block': (P.MaxPool(), {'exception': ValueError, 'error_keywords': ['MaxPool']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 32]).astype(np.float32))],
        'skip': ['backward']}),
    # rank of x is not 4
    ('MaxPool2', {
        'block': (P.MaxPool(kernel_size=50, strides=1), {'exception': ValueError, 'error_keywords': ['MaxPool']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))],
        'skip': ['backward']}),

    # input is scalar
    ('AvgPool0', {
        'block': (P.AvgPool(), {'exception': TypeError, 'error_keywords': ['AvgPool']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # rank of x is not 4
    ('AvgPool1', {
        'block': (P.AvgPool(), {'exception': ValueError, 'error_keywords': ['AvgPool']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 32]).astype(np.float32))],
        'skip': ['backward']}),
    # rank of x is not 4
    ('AvgPool2', {
        'block': (P.AvgPool(kernel_size=50, strides=1), {'exception': ValueError, 'error_keywords': ['AvgPool']}),
        'desc_inputs': [Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))],
        'skip': ['backward']}),

    # input is scalar
    ('Conv2DBackpropInput0', {
        'block': (Conv2DBackpropInputNet(P.Conv2DBackpropInput(2, (5, 5)), (2, 3)),
                  {'exception': TypeError, 'error_keywords': ['Conv2DBackpropInput']}),
        'desc_inputs': [5.0, 5.0],
        'skip': ['backward']}),
    # input is Tensor(bool)
    ('Conv2DBackpropInput1', {
        'block': (Conv2DBackpropInputNet(P.Conv2DBackpropInput(2, (5, 5)), (2, 3)),
                  {'exception': TypeError, 'error_keywords': ['Conv2DBackpropInput']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.bool_)), Tensor(np.ones([5]).astype(np.bool_))],
        'skip': ['backward']}),
    # types of doutput and w mismatch
    ('Conv2DBackpropInput2', {
        'block': (Conv2DBackpropInputNet(P.Conv2DBackpropInput(2, (5, 5)), (2, 3)),
                  {'exception': TypeError, 'error_keywords': ['Conv2DBackpropInput']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.int32)), Tensor(np.ones([5]).astype(np.float32))],
        'skip': ['backward']}),
    # types x_size is not tuple
    ('Conv2DBackpropInput3', {
        'block': (Conv2DBackpropInputNet(P.Conv2DBackpropInput(2, (5, 5)), 2),
                  {'exception': TypeError, 'error_keywords': ['Conv2DBackpropInput']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.int32)), Tensor(np.ones([5]).astype(np.float32))],
        'skip': ['backward']}),
    # types x_size is not tuple(int,...)
    ('Conv2DBackpropInput4', {
        'block': (Conv2DBackpropInputNet(P.Conv2DBackpropInput(2, (5, 5)), (2, 3.0)),
                  {'exception': TypeError, 'error_keywords': ['Conv2DBackpropInput']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.int32)), Tensor(np.ones([5]).astype(np.float32))],
        'skip': ['backward']}),

    # input is scalar
    ('BiasAdd0', {
        'block': (P.BiasAdd(), {'exception': TypeError, 'error_keywords': ['BiasAdd']}),
        'desc_inputs': [5.0, 5.0],
        'skip': ['backward']}),
    # input is Tensor(bool)
    ('BiasAdd1', {
        'block': (P.BiasAdd(), {'exception': TypeError, 'error_keywords': ['BiasAdd']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.bool_)), Tensor(np.ones([5]).astype(np.bool_))],
        'skip': ['backward']}),
    # types of x and bias mismatch
    ('BiasAdd2', {
        'block': (P.BiasAdd(), {'exception': TypeError, 'error_keywords': ['BiasAdd']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.int32)), Tensor(np.ones([5]).astype(np.float32))],
        'skip': ['backward']}),
    # rank of x less than 2
    ('BiasAdd3', {
        'block': (P.BiasAdd(), {'exception': ValueError, 'error_keywords': ['BiasAdd']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.float32)), Tensor(np.ones([5]).astype(np.float32))],
        'skip': ['backward']}),
    # rank of bias is not equal to 1
    ('BiasAdd4', {
        'block': (P.BiasAdd(), {'exception': ValueError, 'error_keywords': ['BiasAdd']}),
        'desc_inputs': [Tensor(np.ones([5, 3]).astype(np.float32)), Tensor(np.ones([5, 3]).astype(np.float32))],
        'skip': ['backward']}),
    # b_shape[0] != x_shape[1]
    ('BiasAdd5', {
        'block': (P.BiasAdd(), {'exception': ValueError, 'error_keywords': ['BiasAdd']}),
        'desc_inputs': [Tensor(np.ones([5, 3]).astype(np.float32)), Tensor(np.ones([5]).astype(np.float32))],
        'skip': ['backward']}),

    # input x is scalar
    ('TopK0', {
        'block': (TopKNet(P.TopK(), 5), {'exception': TypeError, 'error_keywords': ['TopK']}),
        'desc_inputs': [5.0],
        'skip': ['backward']}),
    # input x is Tensor(bool)
    ('TopK1', {
        'block': (TopKNet(P.TopK(), 5), {'exception': TypeError, 'error_keywords': ['TopK']}),
        'desc_inputs': [Tensor(np.ones([10]).astype(np.bool_))],
        'skip': ['backward']}),
    # k is not integer
    ('TopK2', {
        'block': (TopKNet(P.TopK(), 5.0), {'exception': TypeError, 'error_keywords': ['TopK']}),
        'desc_inputs': [Tensor(np.ones([10]).astype(np.float32))],
        'skip': ['backward']}),

    # input is scalar
    ('SoftmaxCrossEntropyWithLogits0', {
        'block': (P.SoftmaxCrossEntropyWithLogits(),
                  {'exception': TypeError, 'error_keywords': ['SoftmaxCrossEntropyWithLogits']}),
        'desc_inputs': [5.0, 5.0],
        'skip': ['backward']}),
    # input is Tensor(bool)
    ('SoftmaxCrossEntropyWithLogits1', {
        'block': (P.SoftmaxCrossEntropyWithLogits(),
                  {'exception': TypeError, 'error_keywords': ['SoftmaxCrossEntropyWithLogits']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.bool_)), Tensor(np.ones([5]).astype(np.bool_))],
        'skip': ['backward']}),
    # types of logits and labels mismatch
    ('SoftmaxCrossEntropyWithLogits2', {
        'block': (P.SoftmaxCrossEntropyWithLogits(),
                  {'exception': TypeError, 'error_keywords': ['SoftmaxCrossEntropyWithLogits']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.float16)), Tensor(np.ones([5]).astype(np.float32))],
        'skip': ['backward']}),
    # shapes of logits and labels mismatch
    ('SoftmaxCrossEntropyWithLogits3', {
        'block': (P.SoftmaxCrossEntropyWithLogits(),
                  {'exception': ValueError, 'error_keywords': ['SoftmaxCrossEntropyWithLogits']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.float32)), Tensor(np.ones([3]).astype(np.float32))],
        'skip': ['backward']}),

    # input is scalar
    ('SparseSoftmaxCrossEntropyWithLogits0', {
        'block': (P.SparseSoftmaxCrossEntropyWithLogits(),
                  {'exception': TypeError, 'error_keywords': ['SparseSoftmaxCrossEntropyWithLogits']}),
        'desc_inputs': [5.0, 5.0],
        'skip': ['backward']}),
    # logits is Tensor(bool)
    ('SparseSoftmaxCrossEntropyWithLogits1', {
        'block': (P.SparseSoftmaxCrossEntropyWithLogits(),
                  {'exception': TypeError, 'error_keywords': ['SparseSoftmaxCrossEntropyWithLogits']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.bool_)), Tensor(np.ones([5]).astype(np.bool_))],
        'skip': ['backward']}),
    # labels is Tensor(bool)
    ('SparseSoftmaxCrossEntropyWithLogits2', {
        'block': (P.SparseSoftmaxCrossEntropyWithLogits(),
                  {'exception': TypeError, 'error_keywords': ['SparseSoftmaxCrossEntropyWithLogits']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.float32)), Tensor(np.ones([5]).astype(np.bool_))],
        'skip': ['backward']}),
    # logits_shape[0] != labels_shape[0]
    ('SparseSoftmaxCrossEntropyWithLogits3', {
        'block': (P.SparseSoftmaxCrossEntropyWithLogits(),
                  {'exception': ValueError, 'error_keywords': ['SparseSoftmaxCrossEntropyWithLogits']}),
        'desc_inputs': [Tensor(np.ones([5]).astype(np.float32)), Tensor(np.ones([3]).astype(np.int32))],
        'skip': ['backward']}),
]


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config_exception)
def test_check_exception():
    return raise_set

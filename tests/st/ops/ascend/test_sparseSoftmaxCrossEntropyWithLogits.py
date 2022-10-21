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
# ============================================================================
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, is_grad=False):
        super(Net, self).__init__()
        self.SparseSoftmaxCrossEntropyWithLogits = P.SparseSoftmaxCrossEntropyWithLogits(is_grad=is_grad)

    @jit
    def construct(self, features, labels):
        return self.SparseSoftmaxCrossEntropyWithLogits(features, labels)


def np_sparse_softmax_cross_entropy_with_logits(labels_shape, logits_shape, logits_dtype):
    num_class = logits_shape[1]
    labels = np.random.randint(low=0, high=num_class - 1, size=labels_shape).astype(np.int32)
    logits = np.random.rand(*logits_shape).astype(logits_dtype)
    features = logits
    features_reshape = np.reshape(features, [-1, num_class])
    labels_reshape = np.reshape(labels, [-1])
    batch_dim = 0
    class_dim = 1
    batch_size = features_reshape.shape[batch_dim]
    e = np.exp(features_reshape - np.reshape(np.amax(features_reshape, axis=class_dim), [batch_size, 1]))
    probs = e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])
    labels_mat = np.zeros_like(probs).astype(probs.dtype)
    labels_mat[np.arange(batch_size), labels_reshape] = 1.0
    bp = (probs - labels_mat)
    loss = -np.sum(labels_mat * np.log(probs + 1.0e-20), axis=1)
    bp_res = np.reshape(bp, features.shape)
    loss_res = np.reshape(loss, labels.shape)
    loss_res = np.sum(loss_res, axis=0) / loss_res.shape[0]
    return labels, logits, loss_res, bp_res


def test_net():
    '''Compare Numpy with MS type is float32'''
    labels_shape = (32,)
    logits_shape = [32, 1001]
    labels, logits, loss_np, _ = np_sparse_softmax_cross_entropy_with_logits(labels_shape, logits_shape, np.float32)
    expect = loss_np
    SparseSoftmaxCrossEntropyWithLogits = Net()
    loss_me = SparseSoftmaxCrossEntropyWithLogits(Tensor(logits), Tensor(labels))
#   assert
    assert np.allclose(expect.flatten(), loss_me.asnumpy().flatten(), 0.01, 0.01)
    print(loss_me.asnumpy().flatten())
    print("-------------------------")
    print(expect)


test_net()

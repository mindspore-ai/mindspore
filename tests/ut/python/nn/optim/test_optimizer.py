# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
""" test optimizer """
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.nn.optim import Optimizer, SGD, Adam, AdamWeightDecay


class IterableObjc:
    def __iter__(self):
        cont = 0
        while cont < 3:
            cont += 1
            yield Parameter(Tensor(cont), name="cont" + str(cont))


params = IterableObjc()


class TestOptimizer():
    def test_init(self):
        Optimizer(0.5, params)
        with pytest.raises(ValueError):
            Optimizer(-0.5, params)

    def test_construct(self):
        opt_2 = Optimizer(0.5, params)
        with pytest.raises(NotImplementedError):
            opt_2.construct()


class TestAdam():
    """ TestAdam definition """

    def test_init(self):
        Adam(params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False,
             use_nesterov=False, weight_decay=0.0, loss_scale=1.0)

    def test_construct(self):
        with pytest.raises(RuntimeError):
            gradient = Tensor(np.zeros([1, 2, 3]))
            adam = Adam(params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False,
                        use_nesterov=False, weight_decay=0.0, loss_scale=1.0)
            adam(gradient)


class TestSGD():
    """ TestSGD definition """

    def test_init(self):
        with pytest.raises(ValueError):
            SGD(params, learning_rate=0.1, momentum=-0.1, dampening=0, weight_decay=0, nesterov=False)
        with pytest.raises(ValueError):
            SGD(params, learning_rate=0.12, momentum=-0.1, dampening=0, weight_decay=0, nesterov=False)
        SGD(params)


class TestNullParam():
    """ TestNullParam definition """

    def test_optim_init(self):
        with pytest.raises(ValueError):
            Optimizer(0.1, None)

    def test_AdamWightDecay_init(self):
        with pytest.raises(ValueError):
            AdamWeightDecay(None)

    def test_Sgd_init(self):
        with pytest.raises(ValueError):
            SGD(None)


class TestUnsupportParam():
    """ TestUnsupportParam definition """

    def test_optim_init(self):
        with pytest.raises(TypeError):
            Optimizer(0.1, (1, 2, 3))

    def test_AdamWightDecay_init(self):
        with pytest.raises(TypeError):
            AdamWeightDecay(9)

    def test_Sgd_init(self):
        with pytest.raises(TypeError):
            paramsTensor = Parameter(Tensor(np.zeros([1, 2, 3])), "x")
            SGD(paramsTensor)


def test_not_flattened_params():
    """
    Feature: Flatten weights.
    Description: Optimizer with not flattened parameters.
    Expectation: The Optimizer works as expected.
    """
    p1 = Parameter(Tensor([1], ms.float32), name="p1")
    p2 = Parameter(Tensor([2], ms.float32), name="p2")
    p3 = Parameter(Tensor([3], ms.float32), name="p3")
    paras = [p1, p2, p3]
    opt = Optimizer(0.1, paras)
    assert not opt._use_flattened_params  # pylint: disable=W0212
    assert len(opt.parameters) == 3
    assert len(opt.cache_enable) == 3


def test_with_flattened_params():
    """
    Feature: Flatten weights.
    Description: Optimizer with flattened parameters.
    Expectation: The Optimizer works as expected.
    """
    p1 = Parameter(Tensor([1], ms.float32), name="p1")
    p2 = Parameter(Tensor([2], ms.float32), name="p2")
    p3 = Parameter(Tensor([3], ms.float32), name="p3")
    paras = [p1, p2, p3]
    Tensor._flatten_tensors(paras)  # pylint: disable=W0212
    opt = Optimizer(0.1, paras)
    assert opt._use_flattened_params  # pylint: disable=W0212
    assert len(opt.parameters) == 1
    assert len(opt.cache_enable) == 1
    assert opt.parameters[0].dtype == ms.float32
    assert opt.parameters[0].shape == [3]
    assert opt.parameters[0]._size == 3  # pylint: disable=W0212
    assert np.allclose(opt.parameters[0].asnumpy(), np.array([1, 2, 3]))
    p1.asnumpy()[0] = 6
    p2.asnumpy()[0] = 6
    p3.asnumpy()[0] = 6
    assert np.allclose(opt.parameters[0].asnumpy(), np.array([6, 6, 6]))


def test_adam_with_flattened_params():
    """
    Feature: Flatten weights.
    Description: Adam optimizer with flattened parameters.
    Expectation: It is ok to compile the optimizer.
    """
    p1 = Parameter(Tensor([1], ms.float32), name="p1")
    p2 = Parameter(Tensor([2], ms.float32), name="p2")
    p3 = Parameter(Tensor([3], ms.float32), name="p3")
    paras = [p1, p2, p3]
    Tensor._flatten_tensors(paras)  # pylint: disable=W0212
    adam = Adam(paras)
    g1 = Tensor([0.1], ms.float32)
    g2 = Tensor([0.2], ms.float32)
    g3 = Tensor([0.3], ms.float32)
    grads = (g1, g2, g3)
    with pytest.raises(NotImplementedError):
        adam(grads)

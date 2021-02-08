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
# ==============================================================================
import pytest
import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.nn.probability.distribution as msd
from mindspore import context, Tensor
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype
from mindspore import dtype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Sampling(nn.Cell):
    """
    Test class: sample of Normal distribution.
    """
    def __init__(self, shape, seed=0):
        super(Sampling, self).__init__()
        self.n1 = msd.Normal(0, 1, seed=seed, dtype=dtype.float32)
        self.shape = shape

    def construct(self, mean=None, sd=None):
        s1 = self.n1.sample(self.shape, mean, sd)
        s2 = self.n1.sample(self.shape, mean, sd)
        s3 = self.n1.sample(self.shape, mean, sd)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sample_graph():
    shape = (2, 3)
    seed = 0
    samp = Sampling(shape, seed=seed)
    sample1, sample2, sample3 = samp()
    assert ((sample1 != sample2).any() and (sample1 != sample3).any() and (sample2 != sample3).any()), \
        "The results should be different!"


class CompositeNormalNet(nn.Cell):
    def __init__(self, shape=None, seed=0):
        super(CompositeNormalNet, self).__init__()
        self.shape = shape
        self.seed = seed

    def construct(self, mean, stddev):
        s1 = C.normal(self.shape, mean, stddev, self.seed)
        s2 = C.normal(self.shape, mean, stddev, self.seed)
        s3 = C.normal(self.shape, mean, stddev, self.seed)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_composite_normal():
    shape = (3, 2, 4)
    mean = Tensor(0.0, mstype.float32)
    stddev = Tensor(1.0, mstype.float32)
    net = CompositeNormalNet(shape)
    s1, s2, s3 = net(mean, stddev)
    assert ((s1 != s2).any() and (s1 != s3).any() and (s2 != s3).any()), \
        "The results should be different!"


class CompositeLaplaceNet(nn.Cell):
    def __init__(self, shape=None, seed=0):
        super(CompositeLaplaceNet, self).__init__()
        self.shape = shape
        self.seed = seed

    def construct(self, mean, lambda_param):
        s1 = C.laplace(self.shape, mean, lambda_param, self.seed)
        s2 = C.laplace(self.shape, mean, lambda_param, self.seed)
        s3 = C.laplace(self.shape, mean, lambda_param, self.seed)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_composite_laplace():
    shape = (3, 2, 4)
    mean = Tensor(1.0, mstype.float32)
    lambda_param = Tensor(1.0, mstype.float32)
    net = CompositeLaplaceNet(shape)
    s1, s2, s3 = net(mean, lambda_param)
    assert ((s1 != s2).any() and (s1 != s3).any() and (s2 != s3).any()), \
        "The results should be different!"


class CompositeGammaNet(nn.Cell):
    def __init__(self, shape=None, seed=0):
        super(CompositeGammaNet, self).__init__()
        self.shape = shape
        self.seed = seed

    def construct(self, alpha, beta):
        s1 = C.gamma(self.shape, alpha, beta, self.seed)
        s2 = C.gamma(self.shape, alpha, beta, self.seed)
        s3 = C.gamma(self.shape, alpha, beta, self.seed)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_composite_gamma():
    shape = (3, 2, 4)
    alpha = Tensor(1.0, mstype.float32)
    beta = Tensor(1.0, mstype.float32)
    net = CompositeGammaNet(shape)
    s1, s2, s3 = net(alpha, beta)
    assert ((s1 != s2).any() and (s1 != s3).any() and (s2 != s3).any()), \
        "The results should be different!"


class CompositePoissonNet(nn.Cell):
    def __init__(self, shape=None, seed=0):
        super(CompositePoissonNet, self).__init__()
        self.shape = shape
        self.seed = seed

    def construct(self, mean):
        s1 = C.poisson(self.shape, mean, self.seed)
        s2 = C.poisson(self.shape, mean, self.seed)
        s3 = C.poisson(self.shape, mean, self.seed)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_composite_poisson():
    shape = (3, 2, 4)
    mean = Tensor(2.0, mstype.float32)
    net = CompositePoissonNet(shape)
    s1, s2, s3 = net(mean)
    assert ((s1 != s2).any() and (s1 != s3).any() and (s2 != s3).any()), \
        "The results should be different!"


class CompositeUniformNet(nn.Cell):
    def __init__(self, shape=None, seed=0):
        super(CompositeUniformNet, self).__init__()
        self.shape = shape
        self.seed = seed

    def construct(self, a, b):
        s1 = C.uniform(self.shape, a, b, self.seed)
        s2 = C.uniform(self.shape, a, b, self.seed)
        s3 = C.uniform(self.shape, a, b, self.seed)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_composite_uniform():
    shape = (3, 2, 4)
    a = Tensor(0.0, mstype.float32)
    b = Tensor(1.0, mstype.float32)
    net = CompositeUniformNet(shape)
    s1, s2, s3 = net(a, b)
    assert ((s1 != s2).any() and (s1 != s3).any() and (s2 != s3).any()), \
        "The results should be different!"


class StandardNormalNet(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(StandardNormalNet, self).__init__()
        self.shape = shape
        self.seed = seed
        self.seed2 = seed2
        self.standard_normal = P.StandardNormal(seed, seed2)

    def construct(self):
        s1 = self.standard_normal(self.shape)
        s2 = self.standard_normal(self.shape)
        s3 = self.standard_normal(self.shape)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_standard_normal():
    shape = (4, 16)
    net = StandardNormalNet(shape)
    s1, s2, s3 = net()
    assert ((s1 != s2).any() and (s1 != s3).any() and (s2 != s3).any()), \
        "The results should be different!"


class StandardLaplaceNet(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(StandardLaplaceNet, self).__init__()
        self.shape = shape
        self.seed = seed
        self.seed2 = seed2
        self.standard_laplace = P.StandardLaplace(seed, seed2)

    def construct(self):
        s1 = self.standard_laplace(self.shape)
        s2 = self.standard_laplace(self.shape)
        s3 = self.standard_laplace(self.shape)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_standard_laplace():
    shape = (4, 16)
    net = StandardLaplaceNet(shape)
    s1, s2, s3 = net()
    assert ((s1 != s2).any() and (s1 != s3).any() and (s2 != s3).any()), \
        "The results should be different!"


class GammaNet(nn.Cell):
    def __init__(self, shape, alpha, beta, seed=0, seed2=0):
        super(GammaNet, self).__init__()
        self.shape = shape
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.seed2 = seed2
        self.gamma = P.Gamma(seed, seed2)

    def construct(self):
        s1 = self.gamma(self.shape, self.alpha, self.beta)
        s2 = self.gamma(self.shape, self.alpha, self.beta)
        s3 = self.gamma(self.shape, self.alpha, self.beta)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_gamma():
    shape = (4, 16)
    alpha = Tensor(1.0, mstype.float32)
    beta = Tensor(1.0, mstype.float32)
    net = GammaNet(shape, alpha, beta)
    s1, s2, s3 = net()
    assert ((s1 != s2).any() and (s1 != s3).any() and (s2 != s3).any()), \
        "The results should be different!"


class PoissonNet(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(PoissonNet, self).__init__()
        self.shape = shape
        self.seed = seed
        self.seed2 = seed2
        self.poisson = P.Poisson(seed, seed2)

    def construct(self, mean):
        s1 = self.poisson(self.shape, mean)
        s2 = self.poisson(self.shape, mean)
        s3 = self.poisson(self.shape, mean)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_poisson():
    shape = (4, 16)
    mean = Tensor(5.0, mstype.float32)
    net = PoissonNet(shape=shape)
    s1, s2, s3 = net(mean)
    assert ((s1 != s2).any() and (s1 != s3).any() and (s2 != s3).any()), \
        "The results should be different!"


class UniformIntNet(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(UniformIntNet, self).__init__()
        self.shape = shape
        self.seed = seed
        self.seed2 = seed2
        self.uniform_int = P.UniformInt(seed, seed2)

    def construct(self, minval, maxval):
        s1 = self.uniform_int(self.shape, minval, maxval)
        s2 = self.uniform_int(self.shape, minval, maxval)
        s3 = self.uniform_int(self.shape, minval, maxval)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_uniform_int():
    shape = (4, 16)
    minval = Tensor(1, mstype.int32)
    maxval = Tensor(5, mstype.int32)
    net = UniformIntNet(shape)
    s1, s2, s3 = net(minval, maxval)
    assert ((s1 != s2).any() and (s1 != s3).any() and (s2 != s3).any()), \
        "The results should be different!"


class UniformRealNet(nn.Cell):
    def __init__(self, shape, seed=0, seed2=0):
        super(UniformRealNet, self).__init__()
        self.shape = shape
        self.seed = seed
        self.seed2 = seed2
        self.uniform_real = P.UniformReal(seed, seed2)

    def construct(self):
        s1 = self.uniform_real(self.shape)
        s2 = self.uniform_real(self.shape)
        s3 = self.uniform_real(self.shape)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_uniform_real():
    shape = (4, 16)
    net = UniformRealNet(shape)
    s1, s2, s3 = net()
    assert ((s1 != s2).any() and (s1 != s3).any() and (s2 != s3).any()), \
        "The results should be different!"


class DropoutGenMaskNet(nn.Cell):
    def __init__(self, shape):
        super(DropoutGenMaskNet, self).__init__()
        self.shape = shape
        self.dropout_gen_mask = P.DropoutGenMask(Seed0=0, Seed1=0)

    def construct(self, keep_prob):
        s1 = self.dropout_gen_mask(self.shape, keep_prob)
        s2 = self.dropout_gen_mask(self.shape, keep_prob)
        s3 = self.dropout_gen_mask(self.shape, keep_prob)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dropout_gen_mask():
    shape = (2, 4, 5)
    keep_prob = Tensor(0.5, mstype.float32)
    net = DropoutGenMaskNet(shape)
    s1, s2, s3 = net(keep_prob)
    assert ((s1 != s2).any() and (s1 != s3).any() and (s2 != s3).any()), \
        "The results should be different!"


class RandomChoiceWithMaskNet(nn.Cell):
    def __init__(self):
        super(RandomChoiceWithMaskNet, self).__init__()
        self.rnd_choice_mask = P.RandomChoiceWithMask(count=4, seed=0)

    def construct(self, x):
        index1, _ = self.rnd_choice_mask(x)
        index2, _ = self.rnd_choice_mask(x)
        index3, _ = self.rnd_choice_mask(x)
        return index1, index2, index3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_random_choice_with_mask():
    net = RandomChoiceWithMaskNet()
    x = Tensor(np.array([[1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1]]).astype(np.bool))
    index1, index2, index3 = net(x)
    assert ((index1 != index2).any() and (index1 != index3).any() and (index2 != index3).any()), \
        "The results should be different!"


class RandomCategoricalNet(nn.Cell):
    def __init__(self, num_sample):
        super(RandomCategoricalNet, self).__init__()
        self.random_categorical = P.RandomCategorical(mstype.int64)
        self.num_sample = num_sample

    def construct(self, logits, seed=0):
        s1 = self.random_categorical(logits, self.num_sample, seed)
        s2 = self.random_categorical(logits, self.num_sample, seed)
        s3 = self.random_categorical(logits, self.num_sample, seed)
        return s1, s2, s3


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_random_categorical():
    num_sample = 8
    net = RandomCategoricalNet(num_sample)
    x = Tensor(np.random.random((10, 5)).astype(np.float32))
    # Outputs may be the same, only basic functions are verified here.
    net(x)

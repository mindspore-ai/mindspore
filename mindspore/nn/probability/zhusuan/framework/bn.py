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
""" Bayesian Network """

import mindspore.nn as nn

import mindspore.nn.probability.distribution as msd
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P


class BayesianNet(nn.Cell):
    """
    We currently support 3 types of variables: x = observation, z = latent, y = condition.
    A Bayeisian Network models a generative process for certain variables: p(x,z|y) or p(z|x,y) or p(x|z,y)
    """

    def __init__(self):
        super().__init__()
        self.normal_dist = msd.Normal(dtype=mstype.float32)
        self.bernoulli_dist = msd.Bernoulli(dtype=mstype.float32)

        self.reduce_sum = P.ReduceSum(keep_dims=True)

    def Normal(self,
               name,
               observation=None,
               mean=None,
               std=None,
               seed=0,
               dtype=mstype.float32,
               shape=(),
               reparameterize=True):
        """ Normal distribution wrapper """

        assert not name is None
        assert not seed is None
        assert not dtype is None

        if observation is None:
            if reparameterize:
                epsilon = self.normal_dist('sample', shape, self.zeros(
                    mean.shape), self.ones(std.shape))
                sample = mean + std * epsilon
            else:
                sample = self.normal_dist('sample', shape, mean, std)
        else:
            sample = observation

        log_prob = self.reduce_sum(self.normal_dist(
            'log_prob', sample, mean, std), 1)
        return sample, log_prob

    def Bernoulli(self,
                  name,
                  observation=None,
                  probs=None,
                  seed=0,
                  dtype=mstype.float32,
                  shape=()):
        """ Bernoulli distribution wrapper """

        assert not name is None
        assert not seed is None
        assert not dtype is None

        if observation is None:
            sample = self.bernoulli_dist('sample', shape, probs)
        else:
            sample = observation

        log_prob = self.reduce_sum(
            self.bernoulli_dist('log_prob', sample, probs), 1)
        return sample, log_prob

    def construct(self, *inputs, **kwargs):
        """
        We currently fix the parameters of the construct function.
        Args:
            the inputs must consist of 3 variables in order.
            x: data sample, observation
            z: latent variable
            y: conditional information
        """
        raise NotImplementedError

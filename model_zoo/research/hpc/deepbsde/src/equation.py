# Copyright 2021 Huawei Technologies Co., Ltd
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
"""equations for different PDE function."""

import numpy as np
from scipy.stats import multivariate_normal as normal
from mindspore import ops as P
from mindspore import nn
import mindspore.dataset as ds

class Equation():
    """Base class for defining PDE related function."""

    def __init__(self, cfg):
        self.dim = cfg.dim
        self.total_time = cfg.total_time
        self.steps = cfg.num_iterations
        self.num_time_interval = cfg.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None
        self.num_sample = cfg.batch_size
        self.generator = P.Identity()
        self.terminal_condition = P.Identity()

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def __getitem__(self, index):
        return self.sample(self.num_sample)

    @property
    def column_names(self):
        return ["dw", "x"]

    def __len__(self):
        return self.steps


class HJBLQ(Equation):
    """HJB equation in PNAS paper doi.org/10.1073/pnas.1718942115"""
    def __init__(self, cfg):
        super(HJBLQ, self).__init__(cfg)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.generator = HJBLQGenerator(1.0)
        self.terminal_condition = HJBLQTerminalCondition()

    def sample(self, num_sample):
        # draw random samples from a multivariate normal distribution
        dw_sample = normal.rvs(size=[num_sample,
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t  # num_sample, dim, num_time_interval
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        for i in range(self.num_time_interval):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sigma * dw_sample[:, :, i]
        return dw_sample.astype(np.float32), x_sample.astype(np.float32)


class HJBLQGenerator(nn.Cell):
    """Generator function for HJBLQ"""
    def __init__(self, lambd):
        super(HJBLQGenerator, self).__init__()
        self.lambd = lambd
        self.sum = P.ReduceSum(keep_dims=True)
        self.square = P.Square()

    def construct(self, t, x, y, z):
        res = -self.lambd * self.sum(self.square(z), 1)
        return res


class HJBLQTerminalCondition(nn.Cell):
    """Terminal condition for HJBLQ"""
    def __init__(self):
        super(HJBLQTerminalCondition, self).__init__()
        self.sum = P.ReduceSum(keep_dims=True)
        self.square = P.Square()

    def construct(self, t, x):
        res = P.log((1 + self.sum(self.square(x), 1)) / 2)
        return res

def get_bsde(cfg):
    bsde_dict = {"HJBLQ": HJBLQ(cfg)}
    return bsde_dict[cfg.eqn_name.upper()]

def create_dataset(bsde):
    """Get generator dataset when training."""
    dataset = ds.GeneratorDataset(bsde, bsde.column_names)
    return dataset

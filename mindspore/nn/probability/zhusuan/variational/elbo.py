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
""" ELBO """

import mindspore.nn as nn

from mindspore.ops import operations as P


class ELBO(nn.Cell):
    """ ELBO class """
    def __init__(self, generator, variational):
        super().__init__()
        self.generator = generator
        self.variational = variational
        self.reshape_op = P.Reshape()
        self.reduce_mean = P.ReduceMean(keep_dims=False)
        self.square = P.Square()

    def construct(self, *inputs, **kwargs):
        if len(inputs) >= 2:
            x, y = inputs[0], inputs[1]
        else:
            x = inputs[0]
            y = None

        z, log_prob_z = self.variational(x, None, y)
        _, log_prob_x_, _, log_prob_z_ = self.generator(x, z, y)

        elbo = self.reduce_mean(log_prob_x_) + self.reduce_mean(log_prob_z_) - self.reduce_mean(log_prob_z)
        return -elbo

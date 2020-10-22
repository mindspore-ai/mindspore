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

from numpy import allclose
from mindspore.common import set_seed
import mindspore.common.initializer as init
import mindspore.nn as nn
from mindspore import Parameter

parameter_shape = [16, 4]


class ParameterNet(nn.Cell):
    def __init__(self):
        super(ParameterNet, self).__init__()
        self.para_xavier_uniform = Parameter(init.initializer('xavier_uniform', parameter_shape), name="xavier_uniform")
        self.para_he_uniform = Parameter(init.initializer('he_uniform', parameter_shape), name="he_uniform")
        self.para_xavier_uniform2 = Parameter(init.initializer(init.XavierUniform(), parameter_shape),
                                              name="xavier_uniform2")
        self.para_he_uniform2 = Parameter(init.initializer(init.HeUniform(), parameter_shape), name="he_uniform2")
        self.para_truncated_normal = Parameter(init.initializer(init.TruncatedNormal(), parameter_shape),
                                               name="truncated_normal")
        self.para_normal = Parameter(init.initializer(init.Normal(), parameter_shape), name="normal")
        self.para_uniform = Parameter(init.initializer(init.Uniform(), parameter_shape), name="uniform")

    def construct(self):
        raise NotImplementedError


def test_using_same_seed_for_initializer():
    set_seed(0)
    net1 = ParameterNet()
    net1.init_parameters_data()
    set_seed(0)
    net2 = ParameterNet()
    net2.init_parameters_data()
    for key in net1.parameters_dict():
        if key not in net2.parameters_dict():
            assert False
        else:
            assert allclose(net1.parameters_dict()[key].data.asnumpy(), net2.parameters_dict()[key].data.asnumpy())


def test_using_diffserent_seed_for_initializer():
    set_seed(0)
    net1 = ParameterNet()
    net1.init_parameters_data()
    set_seed(1)
    net2 = ParameterNet()
    net2.init_parameters_data()
    for key in net1.parameters_dict():
        if key not in net2.parameters_dict():
            assert False
        else:
            assert not allclose(net1.parameters_dict()[key].data.asnumpy(), net2.parameters_dict()[key].data.asnumpy())


if __name__ == '__main__':
    test_using_diffserent_seed_for_initializer()
    test_using_same_seed_for_initializer()

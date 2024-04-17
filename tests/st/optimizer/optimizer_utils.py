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

import numpy as np
import mindspore
from mindspore import nn, Tensor
from mindspore.ops import operations as P
from mindspore.nn.optim import ASGD
from mindspore.nn.optim import Rprop
from mindspore.nn.optim import AdaMax

np.random.seed(1024)

fc1_weight = np.array([[0.72346634, 0.95608497, 0.4084163, 0.18627149,
                        0.6942514, 0.39767185, 0.24918061, 0.4548748],
                       [0.7203382, 0.19086994, 0.76286614, 0.87920564,
                        0.3169892, 0.9462494, 0.62827677, 0.27504718],
                       [0.3544535, 0.2524781, 0.5370583, 0.8313121,
                        0.6670143, 0.0488653, 0.62225235, 0.7546456],
                       [0.17985944, 0.05106374, 0.31064633, 0.4863033,
                        0.848814, 0.5523157, 0.20295663, 0.7213356]]).astype("float32")

fc1_bias = np.array([0.79708564, 0.13728078, 0.66322654, 0.88128525]).astype("float32")

fc2_weight = np.array([[0.8473515, 0.50923985, 0.42287776, 0.29769543]]).astype("float32")

fc2_bias = np.array([0.09996348]).astype("float32")


def make_fake_data():
    """
    make fake data
    """
    data, label = [], []
    for i in range(20):
        data.append(mindspore.Tensor(np.array(np.ones((2, 8)) * i, dtype=np.float32)))
        label.append(mindspore.Tensor(np.array(np.ones((2, 1)) * (i + 1), dtype=np.float32)))
    return data, label


class NetWithLoss(nn.Cell):
    """
    build net with loss
    """

    def __init__(self, network, loss_fn):
        super(NetWithLoss, self).__init__()
        self.network = network
        self.loss = loss_fn

    def construct(self, x, label):
        out = self.network(x)
        loss = self.loss(out, label)
        return loss


class FakeNet(nn.Cell):
    """
    build fake net
    """

    def __init__(self):
        super(FakeNet, self).__init__()
        self.fc1 = nn.Dense(in_channels=8, out_channels=4, weight_init=Tensor(fc1_weight), bias_init=Tensor(fc1_bias))
        self.fc2 = nn.Dense(in_channels=4, out_channels=1, weight_init=Tensor(fc2_weight), bias_init=Tensor(fc2_bias))
        self.relu = nn.ReLU()
        self.reducemean = P.ReduceMean()

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        """
        parameter initialization
        """
        self.init_parameters_data()
        for name, m in self.cells_and_names():
            if name == 'fc1':
                m.weight.set_data(Tensor(fc1_weight))
                m.bias.set_data(Tensor(fc1_bias))
            elif name == 'fc2':
                m.weight.set_data(Tensor(fc2_weight))
                m.bias.set_data(Tensor(fc2_bias))


def build_network(opt_config, net, is_group=None, loss_fn=None):
    """
    Construct training
    """
    if is_group is None:
        is_group = False
    if loss_fn is None:
        loss_fn = nn.L1Loss(reduction='mean')
    losses = []
    networkwithloss = NetWithLoss(net, loss_fn)
    networkwithloss.set_train()

    if is_group:
        fc1_params = list(filter(lambda x: 'fc1' in x.name, networkwithloss.trainable_params()))
        fc2_params = list(filter(lambda x: 'fc1' not in x.name, networkwithloss.trainable_params()))
        if opt_config['name'] == 'ASGD':
            params = [{'params': fc1_params, 'weight_decay': 0.01, 'lr': 0.01}, {'params': fc2_params, 'lr': 0.1}]
        elif opt_config['name'] == 'adamax':
            params = [{'params': fc1_params, 'lr': 0.0018}, {'params': fc2_params, 'lr': 0.0022}]
        elif opt_config['name'] == 'SGD':
            params = [{'params': fc1_params, 'weight_decay': 0.2}, {'params': fc2_params}]
        else:
            params = [{'params': fc1_params, 'lr': 0.01}, {'params': fc2_params, 'lr': 0.01}]
    else:
        params = networkwithloss.trainable_params()

    if opt_config['name'] == 'ASGD':
        net_opt = ASGD(params, learning_rate=opt_config['lr'], lambd=opt_config['lambd'], alpha=opt_config['alpha'],
                       t0=opt_config['t0'], weight_decay=opt_config['weight_decay'])

    elif opt_config['name'] == 'Rprop':
        net_opt = Rprop(params, learning_rate=opt_config['lr'], etas=opt_config['etas'],
                        step_sizes=opt_config['step_sizes'], weight_decay=0.0)

    elif opt_config['name'] == 'adamax':
        net_opt = AdaMax(params, learning_rate=opt_config['lr'], beta1=opt_config['beta1'],
                         beta2=opt_config['beta2'], eps=opt_config['eps'], weight_decay=0.0)
    elif opt_config['name'] == 'SGD':
        net_opt = nn.SGD(params, weight_decay=opt_config['weight_decay'], dampening=0.3, momentum=0.1)
    trainonestepcell = mindspore.nn.TrainOneStepCell(networkwithloss, net_opt)
    data, label = make_fake_data()
    for i in range(20):
        loss = trainonestepcell(data[i], label[i])
        losses.append(loss.asnumpy())
    return np.array(losses), net_opt


default_fc1_weight_asgd = np.array([[0.460443, 0.693057, 0.145399, -0.076741, 0.431228, 0.134655,
                                     -0.013833, 0.191857],
                                    [0.391073, -0.138385, 0.433600, 0.549937, -0.012268, 0.616980,
                                     0.299013, -0.054209],
                                    [0.064144, -0.037829, 0.246745, 0.540993, 0.376698, -0.241438,
                                     0.331937, 0.464328],
                                    [-0.066224, -0.195017, 0.064560, 0.240214, 0.602717, 0.306225,
                                     -0.043127, 0.475241]], dtype=np.float32)
default_fc1_bias_asgd = np.array([0.740427, 0.091827, 0.624849, 0.851911], dtype=np.float32)
default_fc2_weight_asgd = np.array([[0.585555, 0.512303, 0.424419, 0.323499]], dtype=np.float32)
default_fc2_bias_asgd = np.array([0.059962], dtype=np.float32)

no_default_fc1_weight_asgd = np.array([[0.645291, 0.877900, 0.330253, 0.108117, 0.616077, 0.319509, 0.171024,
                                        0.376710],
                                       [0.687056, 0.157610, 0.729583, 0.845918, 0.283724, 0.912958, 0.594999,
                                        0.241783],
                                       [0.328432, 0.226461, 0.511030, 0.805272, 0.640981, 0.022857, 0.596221,
                                        0.728608],
                                       [0.165102, 0.036311, 0.295884, 0.471533, 0.834030, 0.537543, 0.188198,
                                        0.706556]], dtype=np.float32)
no_default_fc1_bias_asgd = np.array([0.785650, 0.131580, 0.658614, 0.878328], dtype=np.float32)
no_default_fc2_weight_asgd = np.array([[0.374859, -0.049370, -0.068307, -0.115195]], dtype=np.float32)
no_default_fc2_bias_asgd = np.array([0.083960], dtype=np.float32)

no_default_group_fc1_weight_asgd = np.array([[0.197470, 0.429578, -0.116887, -0.338544, 0.168320, -0.127608,
                                              -0.275773, -0.070531],
                                             [0.119964, -0.408341, 0.162399, 0.278482, -0.282498, 0.345379,
                                              0.028105, -0.324348],
                                             [-0.168310, -0.270062, 0.013893, 0.307500, 0.143563, -0.473227,
                                              0.098900, 0.231002],
                                             [-0.254349, -0.382861, -0.123849, 0.051422, 0.413136, 0.117289,
                                              -0.231302, 0.285938]], dtype=np.float32)
no_default_group_fc1_bias_asgd = np.array([0.706595, 0.042866, 0.579553, 0.811499], dtype=np.float32)
no_default_group_fc2_weight_asgd = np.array([[-0.076689, -0.092399, -0.072100, -0.054189]], dtype=np.float32)
no_default_group_fc2_bias_asgd = np.array([0.698678], dtype=np.float32)

default_fc1_weight_sgd = np.array([[0.00533873, 0.03210080, -0.03090680, -0.05646387, 0.00197765,
                                    -0.03214293, -0.04922638, -0.02556189],
                                   [-0.00658702, -0.06750072, -0.00169432, 0.01169018, -0.05299109,
                                    0.01940336, -0.01717841, -0.05781638],
                                   [-0.03723934, -0.04897130, -0.01623122, 0.01762178, -0.00128018,
                                    -0.07239634, -0.00642990, 0.00880153],
                                   [-0.04421479, -0.05903235, -0.02916817, -0.00895938, 0.03274637,
                                    -0.00136485, -0.04155754, 0.01808037]], dtype=np.float32)
default_fc2_weight_sgd = np.array([[-0.01070179, -0.00702989, -0.00210839, 0.00160410]], dtype=np.float32)

default_fc1_weight_adamax = np.array([[0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                       0.00000000, 0.00000000, 0.00000000],
                                      [11.18415642, 11.18415642, 11.18415642, 11.18415642, 11.18415642,
                                       11.18415642, 11.18415642, 11.18415642],
                                      [-6.70855522, -6.70855522, -6.70855522, -6.70855522, -6.70855522,
                                       -6.70855522, -6.70855522, -6.70855522],
                                      [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                       0.00000000, 0.00000000, 0.00000000]], dtype=np.float32)
default_fc1_bias_adamax = np.array([0.00000000, 0.86349380, -0.51633584, 0.00000000], dtype=np.float32)

no_default_fc1_weight_adamax = np.array([[0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                          0.00000000, 0.00000000, 0.00000000],
                                         [-4.02891350, -4.02891350, -4.02891350, -4.02891350, -4.02891350,
                                          -4.02891350, -4.02891350, -4.02891350],
                                         [3.10859227, 3.10859227, 3.10859227, 3.10859227, 3.10859227,
                                          3.10859227, 3.10859227, 3.10859227],
                                         [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                          0.00000000, 0.00000000, 0.00000000]], dtype=np.float32)
no_default_fc1_bias_adamax = np.array([0.00000000, -0.04809491, 0.06205747, 0.00000000], dtype=np.float32)

default_group_fc1_weight_adamax = np.array([[0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                             0.00000000, 0.00000000, 0.00000000],
                                            [11.07278919, 11.07278919, 11.07278919, 11.07278919, 11.07278919,
                                             11.07278919, 11.07278919, 11.07278919],
                                            [-6.81674862, -6.81674862, -6.81674862, -6.81674862, -6.81674862,
                                             -6.81674862, -6.81674862, -6.81674862],
                                            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                             0.00000000, 0.00000000, 0.00000000]], dtype=np.float32)
default_group_fc1_bias_adamax = np.array([0.00000000, 0.85614461, -0.52348828, 0.00000000], dtype=np.float32)

default_fc1_weight_rprop = np.array([[9.10877514, 9.10877514, 9.10877514, 9.10877514, 9.10877514,
                                      9.10877514, 9.10877514, 9.10877514],
                                     [2.68465400, 2.68465400, 2.68465400, 2.68465400, 2.68465400,
                                      2.68465400, 2.68465400, 2.68465400],
                                     [1.04377401, 1.04377401, 1.04377401, 1.04377401, 1.04377401,
                                      1.04377401, 1.04377401, 1.04377401],
                                     [-1.33468997, -1.33468997, -1.33468997, -1.33468997, -1.33468997,
                                      -1.33468997, -1.33468997, -1.33468997]], dtype=np.float32)
default_fc1_bias_rprop = np.array([0.47940922, 0.14129758, 0.05493547, -0.07024684], dtype=np.float32)

no_default_fc1_weight_rprop = np.array([[8.41605091, 8.41605091, 8.41605091, 8.41605091, 8.41605091, 8.41605091,
                                         8.41605091, 8.41605091],
                                        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                         0.00000000, 0.00000000],
                                        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                         0.00000000, 0.00000000],
                                        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                         0.00000000, 0.00000000]], dtype=np.float32)
no_default_fc1_bias_rprop = np.array([0.44295004, 0.00000000, 0.00000000, 0.00000000], dtype=np.float32)

default_group_fc1_weight_rprop = np.array([[8.41605091, 8.41605091, 8.41605091, 8.41605091, 8.41605091, 8.41605091,
                                            8.41605091, 8.41605091],
                                           [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                            0.00000000, 0.00000000],
                                           [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                            0.00000000, 0.00000000],
                                           [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                                            0.00000000, 0.00000000]], dtype=np.float32)
default_group_fc1_bias_rprop = np.array([0.44295004, 0.00000000, 0.00000000, 0.00000000], dtype=np.float32)

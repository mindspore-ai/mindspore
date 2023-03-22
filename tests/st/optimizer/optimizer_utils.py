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


def build_network(opt_config, net, is_group=False, loss_fn=nn.MSELoss(reduction='sum')):
    """
    Construct training
    """
    losses = []

    networkwithloss = NetWithLoss(net, loss_fn)
    networkwithloss.set_train()

    if is_group:
        fc1_params = list(filter(lambda x: 'fc1' in x.name, networkwithloss.trainable_params()))
        fc2_params = list(filter(lambda x: 'fc1' not in x.name, networkwithloss.trainable_params()))
        if opt_config['name'] == 'ASGD':
            params = [{'params': fc1_params, 'weight_decay': 0.01, 'lr': 0.001}, {'params': fc2_params, 'lr': 0.1}]
        elif opt_config['name'] == 'adamax':
            params = [{'params': fc1_params, 'lr': 0.0018}, {'params': fc2_params, 'lr': 0.0022}]
        elif opt_config['name'] == 'SGD':
            params = [{'params': fc1_params, 'weight_decay': 0.2}, {'params': fc2_params}]
        else:
            params = [{'params': fc1_params, 'lr': 0.001}, {'params': fc2_params, 'lr': 0.1}]
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
    if opt_config['name'] == 'ASGD' or opt_config['name'] == 'SGD':
        return np.array(losses), net_opt
    return np.array(losses)


loss_default_asgd = np.array([3.01246792e-01, 1.20041794e+02, 1.38681079e+03, 2.01250820e+01,
                              3.27283554e+01, 4.76963005e+01, 6.47094269e+01, 8.34786530e+01,
                              1.03742706e+02, 1.25265739e+02, 1.47835190e+02, 1.71259613e+02,
                              1.95367035e+02, 2.20003204e+02, 2.45029831e+02, 2.70323456e+02,
                              2.95774048e+02, 3.21283752e+02, 3.46765594e+02, 3.72143097e+02], dtype=np.float32)

loss_not_default_asgd = np.array([3.01246792e-01, 1.26019104e+02, 1.90600449e+02, 9.70605755e+00,
                                  2.98419113e+01, 3.68430023e+02, 1.06318066e+04, 1.35017746e+02,
                                  1.68673813e+02, 2.05914215e+02, 2.46694992e+02, 2.90972443e+02,
                                  3.38703430e+02, 3.89845123e+02, 4.44355103e+02, 5.02191406e+02,
                                  5.63312500e+02, 6.27676941e+02, 6.95244202e+02, 7.65973816e+02], dtype=np.float32)

loss_group_asgd = np.array([3.01246792e-01, 7.26708527e+01, 2.84905312e+05, 4.17499258e+04,
                            1.46797949e+04, 5.07966602e+03, 1.70935132e+03, 5.47094910e+02,
                            1.59216995e+02, 3.78818207e+01, 5.18196869e+00, 2.62275129e-03,
                            2.09768796e+00, 5.23108435e+00, 7.78943682e+00, 9.57108879e+00,
                            1.07310610e+01, 1.14618425e+01, 1.19147835e+01, 1.21936722e+01], dtype=np.float32)


loss_default_rprop = np.array([3.01246792e-01, 1.19871742e+02, 4.13467163e+02, 8.09146179e+02,
                               1.22364807e+03, 1.56787573e+03, 1.75733594e+03, 1.72866272e+03,
                               1.46183936e+03, 1.00406335e+03, 4.84076874e+02, 9.49734650e+01,
                               2.00592804e+01, 1.87920704e+01, 1.53733969e+01, 1.85836582e+01,
                               5.21527790e-02, 2.01522671e-02, 7.19913816e+00, 8.52459526e+00], dtype=np.float32)

loss_not_default_rprop = np.array([3.0124679e-01, 1.2600269e+02, 4.7351608e+02, 1.0220379e+03,
                                   1.7181555e+03, 2.4367019e+03, 2.9170872e+03, 2.7243464e+03,
                                   1.4999669e+03, 7.5820435e+01, 1.0590715e+03, 5.4336096e+02,
                                   7.0162407e+01, 8.2754419e+02, 9.6329260e+02, 3.4475109e+01,
                                   5.3843134e+02, 6.0064526e+02, 1.1046149e+02, 3.5530117e+03], dtype=np.float32)

loss_group_rprop = np.array([3.0124679e-01, 7.1360558e+01, 4.8910957e+01, 2.1730331e+02,
                             3.0747052e+02, 5.2734237e+00, 5.6865869e+00, 1.7116127e+02,
                             2.0539343e+02, 2.2993685e+01, 2.6194101e+02, 2.8772815e+02,
                             2.4236647e+01, 3.9299741e+02, 3.5600668e+02, 1.4759110e+01,
                             7.2244568e+02, 8.1952783e+02, 9.8913864e+01, 1.1141744e+03], dtype=np.float32)

loss_default_adamax = np.array([1.0, 4.542382, 10.5303135, 18.87176, 29.475002,
                                42.2471, 57.09358, 73.917595, 92.62038, 113.10096,
                                135.25633, 158.9815, 184.16951, 210.71207, 238.49873,
                                267.41818, 297.35782, 328.20422, 359.84293, 392.15878], dtype=np.float32)

loss_not_default_adamax = np.array([1.0, 4.5040994, 9.420462, 14.951918, 20.390736,
                                    25.111732, 28.57695, 30.347034, 30.098299, 27.647425,
                                    22.994541, 16.402872, 8.979612, 2.7966619, 0.025522191,
                                    1.9826386, 8.12521, 15.100327, 18.94126, 19.657328], dtype=np.float32)

loss_group_adamax = np.array([1.0, 4.537268, 10.415594, 18.463926, 28.51337,
                              40.394474, 53.936195, 68.9657, 85.307945, 102.78646,
                              121.22308, 140.4386, 160.25333, 180.48737, 200.96124,
                              221.49626, 241.91531, 262.0436, 281.70914, 300.7426], dtype=np.float32)


default_fc1_weight_asgd = np.array([[-0.9451941, -0.71258026, -1.2602371, -1.4823773,
                                     -0.974408, -1.2709816, -1.4194703, -1.2137808],
                                    [-1.5341775, -2.0636342, -1.4916497, -1.3753126,
                                     -1.9375193, -1.308271, -1.6262367, -1.9794592],
                                    [-1.9886293, -2.0906024, -1.8060291, -1.5117803,
                                     -1.6760755, -2.2942104, -1.7208353, -1.5884445],
                                    [-2.071215, -2.2000103, -1.9404325, -1.7647781,
                                     -1.4022746, -1.6987679, -2.0481179, -1.5297506]], dtype=np.float32)
default_fc1_bias_asgd = np.array([-0.17978168, -1.0764512, -0.578816, -0.2928958], dtype=np.float32)
default_fc2_weight_asgd = np.array([[4.097412, 6.2694297, 5.9203916, 5.3845487]], dtype=np.float32)
default_fc2_bias_asgd = np.array([6.904814], dtype=np.float32)


no_default_fc1_weight_asgd = np.array([[-1.3406217, -1.1080127, -1.655658, -1.8777936,
                                        -1.3698348, -1.6664025, -1.8148884, -1.6092018],
                                       [-1.1475986, -1.6770473, -1.1050745, -0.98873824,
                                        -1.5509329, -0.9216978, -1.2396574, -1.5928726],
                                       [-1.2329121, -1.334883, -1.050313, -0.756071,
                                        -0.92036265, -1.5384867, -0.96512324, -0.8327349],
                                       [-1.0685704, -1.1973612, -0.9377885, -0.7621386,
                                        -0.39964262, -0.69612867, -1.0454736, -0.52711576]], dtype=np.float32)
no_default_fc1_bias_asgd = np.array([0.41264832, -0.19961096, 0.37743938, 0.65807366], dtype=np.float32)
no_default_fc2_weight_asgd = np.array([[-5.660916, -5.9415145, -5.1402636, -4.199707]], dtype=np.float32)
no_default_fc2_bias_asgd = np.array([0.5082278], dtype=np.float32)


no_default_group_fc1_weight_asgd = np.array([[-32.526627, -32.29401, -32.8416, -33.06367, -32.55584,
                                              -32.852345, -33.000767, -32.795143],
                                             [-33.164936, -33.69432, -33.12241, -33.006073, -33.568207,
                                              -32.9391, -33.256996, -33.61015],
                                             [-33.118973, -33.220943, -32.936436, -32.642193, -32.806488,
                                              -33.424484, -32.85125, -32.718857],
                                             [-30.155754, -30.284513, -30.025005, -29.849358, -29.486917,
                                              -29.783375, -30.132658, -29.614393]], dtype=np.float32)
no_default_group_fc1_bias_asgd = np.array([-15.838092, -16.811989, -16.078112, -14.289094], dtype=np.float32)
no_default_group_fc2_weight_asgd = np.array([[1288.7146, 1399.3041, 1292.8445, 1121.4629]], dtype=np.float32)
no_default_group_fc2_bias_asgd = np.array([18.513494], dtype=np.float32)

default_fc1_weight_sgd = np.array([[-6.6273242e-02, -3.9511207e-02, -1.0251881e-01, -1.2807587e-01,
                                    -6.9634348e-02, -1.0375493e-01, -1.2083838e-01, -9.7173907e-02],
                                   [-1.8068390e-02, -7.8982085e-02, -1.3175679e-02, 2.0881524e-04,
                                    -6.4472459e-02, 7.9219900e-03, -2.8659783e-02, -6.9297753e-02],
                                   [-2.5218798e-02, -3.6950763e-02, -4.2106784e-03, 2.9642319e-02,
                                    1.0740350e-02, -6.0375791e-02, 5.5906363e-03, 2.0822065e-02],
                                   [-1.1401306e+01, -1.1416125e+01, -1.1386261e+01, -1.1366054e+01,
                                    -1.1324347e+01, -1.1358459e+01, -1.1398650e+01, -1.1339014e+01]], dtype=np.float32)
default_fc2_weight_sgd = np.array([[-0.5055597, -0.5255496, -0.52437556, 1.0779992]], dtype=np.float32)

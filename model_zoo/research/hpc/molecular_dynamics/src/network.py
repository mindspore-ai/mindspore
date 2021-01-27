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
"""The construction of network for molecular dynamics."""
from copy import deepcopy
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from .descriptor import DescriptorSeA

natoms = [192, 192, 64, 128]
rcut_a = -1
rcut_r = 6.0
rcut_r_smth = 5.8
sel_a = [46, 92]
sel_r = [0, 0]
ntypes = len(sel_a)
nnei_a = 138
nnei_r = 0
nnei = nnei_a + nnei_r
ndescrpt_a = nnei_a * 4
ndescrpt_r = nnei_r * 1
ndescrpt = ndescrpt_a + ndescrpt_r
filter_neuron = [25, 50, 100]
n_axis_neuron = 16
dim_descrpt = filter_neuron[-1] * 16
n_neuron = [240, 240, 240]
type_bias_ae = [-93.57, -187.15]


class MDNet(nn.Cell):
    """MD simulation network."""
    def __init__(self):
        super(MDNet, self).__init__()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.concat0 = P.Concat(axis=0)
        self.tanh = nn.Tanh()
        self.mat = P.MatMul()
        self.batchmat = nn.MatMul()
        self.batchmat_tran = nn.MatMul(transpose_x1=True)
        self.idt1 = Parameter(Tensor(np.random.normal(0.1, 0.001, (240,)), dtype=mstype.float32), name="type0_idt1")
        self.idt2 = Parameter(Tensor(np.random.normal(0.1, 0.001, (240,)), dtype=mstype.float32), name="type0_idt2")
        self.idt3 = Parameter(Tensor(np.random.normal(0.1, 0.001, (240,)), dtype=mstype.float32), name="type1_idt1")
        self.idt4 = Parameter(Tensor(np.random.normal(0.1, 0.001, (240,)), dtype=mstype.float32), name="type1_idt2")
        self.idt = [self.idt1, self.idt2, self.idt3, self.idt4]
        self.neuron = [dim_descrpt] + n_neuron
        self.par = [1] + filter_neuron
        self.process = Processing()
        fc = []
        for i in range(3):
            fc.append(nn.Dense(self.par[i], self.par[i + 1],
                               weight_init=Tensor(np.random.normal(0.0, 1.0 / np.sqrt(self.par[i] + self.par[i + 1]),
                                                                   (self.par[i + 1], self.par[i])),
                                                  dtype=mstype.float32),
                               bias_init=Tensor(np.random.normal(0.0, 1.0, (self.par[i + 1],)), dtype=mstype.float32)))
        for i in range(1, 3):
            fc.append(nn.Dense(self.par[i], self.par[i + 1],
                               weight_init=Tensor(np.random.normal(0.0, 1.0 / np.sqrt(self.par[i] + self.par[i + 1]),
                                                                   (self.par[i + 1], self.par[i])),
                                                  dtype=mstype.float32),
                               bias_init=Tensor(np.random.normal(0.0, 1.0, (self.par[i + 1],)), dtype=mstype.float32)))
        for i in range(3):
            fc.append(nn.Dense(self.par[i], self.par[i + 1],
                               weight_init=Tensor(np.random.normal(0.0, 1.0 / np.sqrt(self.par[i] + self.par[i + 1]),
                                                                   (self.par[i + 1], self.par[i])),
                                                  dtype=mstype.float32),
                               bias_init=Tensor(np.random.normal(0.0, 1.0, (self.par[i + 1],)), dtype=mstype.float32)))
        for i in range(1, 3):
            fc.append(nn.Dense(self.par[i], self.par[i + 1],
                               weight_init=Tensor(np.random.normal(0.0, 1.0 / np.sqrt(self.par[i] + self.par[i + 1]),
                                                                   (self.par[i + 1], self.par[i])),
                                                  dtype=mstype.float32),
                               bias_init=Tensor(np.random.normal(0.0, 1.0, (self.par[i + 1],)), dtype=mstype.float32)))
        self.fc = nn.CellList(fc)
        self.fc0 = deepcopy(self.fc)
        self.fc2 = [self.fc, self.fc0]

        fc = []
        for i in range(3):
            fc.append(nn.Dense(self.neuron[i], self.neuron[i + 1],
                               weight_init=Tensor(
                                   np.random.normal(0.0, 1.0 / np.sqrt(self.neuron[i] + self.neuron[i + 1]),
                                                    (self.neuron[i + 1], self.neuron[i])), dtype=mstype.float32),
                               bias_init=Tensor(np.random.normal(0.0, 1.0, (self.neuron[i + 1],)),
                                                dtype=mstype.float32)))
        fc.append(nn.Dense(240, 1,
                           weight_init=Tensor(
                               np.random.normal(0.0, 1.0 / np.sqrt(240 + 1), (1, 240)), dtype=mstype.float32),
                           bias_init=Tensor(np.random.normal(type_bias_ae[0], 1.0, (1,)), dtype=mstype.float32)))
        for i in range(3):
            fc.append(nn.Dense(self.neuron[i], self.neuron[i + 1],
                               weight_init=Tensor(
                                   np.random.normal(0.0, 1.0 / np.sqrt(self.neuron[i] + self.neuron[i + 1]),
                                                    (self.neuron[i + 1], self.neuron[i])), dtype=mstype.float32),
                               bias_init=Tensor(np.random.normal(0.0, 1.0, (self.neuron[i + 1],)),
                                                dtype=mstype.float32)))
        fc.append(nn.Dense(240, 1,
                           weight_init=Tensor(
                               np.random.normal(0.0, 1.0 / np.sqrt(240 + 1), (1, 240)), dtype=mstype.float32),
                           bias_init=Tensor(np.random.normal(type_bias_ae[1], 1.0, (1,)), dtype=mstype.float32)))
        self.fc1 = nn.CellList(fc)

        xyz_A = np.vstack((np.identity(46), np.zeros([92, 46])))
        self.xyz_A = Tensor(np.reshape(xyz_A, (1, 138, 46)))
        xyz_B = np.vstack((np.zeros([46, 92]), np.identity(92)))
        self.xyz_B = Tensor(np.reshape(xyz_B, (1, 138, 92)))

        xyz_2 = np.vstack((np.identity(n_axis_neuron), np.zeros([self.par[-1] - n_axis_neuron, n_axis_neuron])))
        self.xyz_2 = Tensor(xyz_2)

    def _filter(self, slice_0, slice_1, inputs, fc):
        """filter method."""
        shape = self.shape(inputs)
        slice_inputs = (slice_0, slice_1)
        xyz_scatter_total = []
        for type_i in range(2):
            xyz_scatter = slice_inputs[type_i]
            shape_i = self.shape(xyz_scatter)
            xyz_scatter = self.reshape(xyz_scatter, (-1, 1))
            xyz_scatter = self.tanh(fc[type_i * 5 + 0](xyz_scatter))
            hidden = self.tanh(fc[type_i * 5 + 1](xyz_scatter))
            xyz_scatter = fc[type_i * 5 + 3](xyz_scatter) + hidden
            hidden = self.tanh(fc[type_i * 5 + 2](xyz_scatter))
            xyz_scatter = fc[type_i * 5 + 4](xyz_scatter) + hidden
            xyz_scatter = self.reshape(xyz_scatter, (-1, shape_i[1], 100))
            xyz_scatter_total.append(xyz_scatter)
        xyz_scatter = self.batchmat(self.xyz_A, xyz_scatter_total[0]) + self.batchmat(self.xyz_B, xyz_scatter_total[1])
        xyz_scatter_1 = self.batchmat_tran(inputs, xyz_scatter)
        xyz_scatter_1 = xyz_scatter_1 * (4.0 / (shape[1] * shape[2]))
        xyz_scatter_2 = self.batchmat(xyz_scatter_1, self.xyz_2)
        result = self.batchmat_tran(xyz_scatter_1, xyz_scatter_2)
        return result

    def _fitting(self, slice0, slice1, h, slice2, slice3, o):
        """fitting method."""
        l_layer = []
        slice_data = (slice0, slice1, h, slice2, slice3, o)
        for type_i in range(2):
            layer = self._filter(slice_data[3 * type_i], slice_data[3 * type_i + 1], slice_data[3 * type_i + 2],
                                 self.fc2[type_i])
            layer = self.reshape(layer, (-1, dim_descrpt))
            layer = self.tanh(self.fc1[type_i * 4 + 0](layer))
            layer = layer + self.tanh(self.fc1[type_i * 4 + 1](layer)) * self.idt[2 * type_i + 0]
            layer = layer + self.tanh(self.fc1[type_i * 4 + 2](layer)) * self.idt[2 * type_i + 1]
            final_layer = self.fc1[type_i * 4 + 3](layer)
            l_layer.append(final_layer)
        outs = self.concat0((l_layer[0], l_layer[1]))
        return self.reshape(outs, (-1,))

    def construct(self, inputs):
        """construct function."""
        slice0, slice1, h, slice2, slice3, o = self.process(inputs)
        dout = self._fitting(slice0, slice1, h, slice2, slice3, o)
        return dout


class Processing(nn.Cell):
    """data process."""
    def __init__(self):
        super(Processing, self).__init__()
        self.slice = P.Slice()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.batchmat = nn.MatMul()
        self.split = P.Split(1, 3)
        self.concat = P.Concat(axis=1)
        slice_64 = Tensor(np.hstack((np.identity(64), np.zeros([64, 128]))))
        slice_128 = Tensor(np.hstack((np.zeros([128, 64]), np.identity(128))))
        self.slice_0 = [slice_64, slice_128]
        slice_46 = Tensor(np.hstack((np.identity(46), np.zeros([46, 92]))))
        slice_92 = Tensor(np.hstack((np.zeros([92, 46]), np.identity(92))))
        self.slice_1 = [slice_46, slice_92]
        slice_2 = np.vstack((np.identity(1), np.zeros([3, 1])))
        self.slice_2 = Tensor(slice_2)

    def construct(self, inputs):
        """construct function."""
        slice_data = []
        split_tensor = self.split(inputs)
        split_64 = self.reshape(split_tensor[0], (-1, 138, 4))
        split_128 = self.reshape(self.concat((split_tensor[1], split_tensor[2])), (-1, 138, 4))
        split_t = (split_64, split_128)
        for type_i in range(2):
            for type_j in range(2):
                inputs_reshape = self.batchmat(self.slice_1[type_j], split_t[type_i])
                xyz_scatter = self.batchmat(inputs_reshape, self.slice_2)
                slice_data.append(xyz_scatter)
            slice_data.append(split_t[type_i])
        slice0, slice1, h, slice2, slice3, o = slice_data[0], slice_data[1], slice_data[2], \
                                               slice_data[3], slice_data[4], slice_data[5]
        return slice0, slice1, h, slice2, slice3, o


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True)
        self.network = network

    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout


class Network(nn.Cell):
    """The network to calculate energy, force and virial."""
    def __init__(self):
        super(Network, self).__init__()
        self.reshape = P.Reshape()
        self.sum = P.ReduceSum()
        self.mdnet = MDNet()
        self.grad = Grad(self.mdnet)
        self.descrpt_se_a = DescriptorSeA()
        self.process = Processing()

    def construct(self, d_coord, d_nlist, frames, avg, std, atype, nlist):
        """construct function."""
        _, descrpt, _ = \
            self.descrpt_se_a(d_coord, d_nlist, frames, avg, std, atype)
        # calculate energy and atom_ener
        atom_ener = self.mdnet(descrpt)
        energy_raw = atom_ener
        energy_raw = self.reshape(energy_raw, (-1, natoms[0]))
        energy = self.sum(energy_raw, 1)
        # grad of atom_ener
        net_deriv = self.grad(descrpt)
        return energy, atom_ener, net_deriv

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
"""
Network config setting
"""


# config for Schrodinger equation scenario
config_Sch = {'epoch': 50000, 'lr': 0.0001, 'N0': 50, 'Nb': 50, 'Nf': 20000, 'num_neuron': 100,
              'seed': 2, 'path': './Data/NLS.mat', 'ck_path': './ckpoints/'}

# config for Navier-Stokes equation scenario
config_navier = {'epoch': 19000, 'lr': 0.01, 'n_train': 5000, 'path': './Data/cylinder_nektar_wake.mat',
                 'noise': 0.0, 'num_neuron': 20, 'ck_path': './navier_ckpoints/', 'seed': 1, 'batch_size': 500}

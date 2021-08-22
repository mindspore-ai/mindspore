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
'''LagevinLiuJian'''
import math

import numpy as np


class Langevin_Liujian:
    '''LagevinLiuJian'''
    def __init__(self, controller, atom_numbers):
        self.atom_numbers = atom_numbers
        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)

        self.CONSTANT_TIME_CONVERTION = 20.455
        self.CONSTANT_kB = 0.00198716

        self.target_temperature = 300.0 if "target_temperature" not in controller.Command_Set else float(
            controller.Command_Set["target_temperature"])
        self.gamma_ln = 1.0 if "langevin_gamma" not in controller.Command_Set else float(
            controller.Command_Set["langevin_gamma"])
        self.rand_seed = 1 if "langevin_seed" not in controller.Command_Set else float(
            controller.Command_Set["langevin_seed"])
        self.max_velocity = 10000.0 if "velocity_max" not in controller.Command_Set else float(
            controller.Command_Set["velocity_max"])
        assert self.max_velocity > 0
        print("target temperature is ", self.target_temperature)
        print("friction coefficient is ", self.gamma_ln, "ps^-1")
        print("random seed is ", self.rand_seed)
        self.dt = float(controller.Command_Set["dt"])
        self.dt *= self.CONSTANT_TIME_CONVERTION
        self.half_dt = 0.5 * self.dt
        self.rand_state = np.float32(np.zeros([math.ceil(3 * self.atom_numbers / 4.0) * 16,]))
        self.gamma_ln = self.gamma_ln / self.CONSTANT_TIME_CONVERTION
        self.exp_gamma = math.exp(-1 * self.gamma_ln * self.dt)
        self.sqrt_gamma = math.sqrt((1. - self.exp_gamma * self.exp_gamma) * self.target_temperature * self.CONSTANT_kB)
        self.h_sqrt_mass = [0] * self.atom_numbers
        for i in range(self.atom_numbers):
            self.h_sqrt_mass[i] = self.sqrt_gamma * math.sqrt(1. / self.h_mass[i])

    def read_information_from_amberfile(self, file_path):
        '''read amber file'''
        file = open(file_path, 'r')
        context = file.readlines()
        file.close()
        self.h_mass = [0] * self.atom_numbers
        for idx, val in enumerate(context):
            if "%FLAG MASS" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.atom_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(float, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)

                for i in range(self.atom_numbers):
                    self.h_mass[i] = information[i]
                break

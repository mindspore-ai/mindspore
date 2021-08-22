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
        self.module_name = "langevin_liu"
        self.atom_numbers = atom_numbers
        self.h_mass = []
        print("START INITIALIZING LANGEVIN_LIU DYNAMICS:")
        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)
        else:
            self.read_mass_file(controller)
        self.CONSTANT_TIME_CONVERTION = 20.455
        self.CONSTANT_kB = 0.00198716

        self.target_temperature = 300.0 if "target_temperature" not in controller.Command_Set else float(
            controller.Command_Set["target_temperature"])
        self.gamma_ln = 1.0
        if "gamma" in controller.Command_Set:
            self.gamma_ln = float(controller.Command_Set["gamma"])
        if "langevin_liu_gamma" in controller.Command_Set:
            self.gamma_ln = float(controller.Command_Set["langevin_liu_gamma"])
        print("    langevin_liu_gamma is ", self.gamma_ln)

        self.random_seed = 1 if "seed" not in controller.Command_Set else int(
            controller.Command_Set["seed"])

        print("    target temperature is {} K".format(self.target_temperature))
        print("    friction coefficient is {} ps^-1".format(self.gamma_ln))
        print("    random seed is ", self.random_seed)
        self.dt = 0.001 if "dt" not in controller.Command_Set else float(
            controller.Command_Set["dt"]) * self.CONSTANT_TIME_CONVERTION
        self.half_dt = 0.5 * self.dt

        self.float4_numbers = math.ceil(3 * self.atom_numbers / 4.0)
        self.rand_state = np.float32(np.zeros([self.float4_numbers * 16,]))
        self.gamma_ln = self.gamma_ln / self.CONSTANT_TIME_CONVERTION
        self.exp_gamma = math.exp(-1 * self.gamma_ln * self.dt)
        self.sqrt_gamma = math.sqrt((1. - self.exp_gamma * self.exp_gamma) * self.target_temperature * self.CONSTANT_kB)
        self.h_sqrt_mass = [0] * self.atom_numbers
        for i in range(self.atom_numbers):
            self.h_sqrt_mass[i] = self.sqrt_gamma * math.sqrt(1. / self.h_mass[i]) if self.h_mass[i] != 0 else 0

        self.max_velocity = 0
        if "velocity_max" in controller.Command_Set:
            self.max_velocity = float(controller.Command_Set["velocity_max"])
        if "langevin_liu_velocity_max" in controller.Command_Set:
            self.max_velocity = float(controller.Command_Set["langevin_liu_velocity_max"])
        print("    max velocity is ", self.max_velocity)

        self.h_mass_inverse = [0] * self.atom_numbers
        for i in range(self.atom_numbers):
            self.h_mass_inverse[i] = 1. / self.h_mass[i] if self.h_mass[i] != 0 else 0

        self.is_initialized = 1

        print("END INITIALIZING LANGEVIN_LIU DYNAMICS")

    def read_mass_file(self, controller):
        if "mass_in_file" in controller.Command_Set:
            path = controller.Command_Set["mass_in_file"]
            file = open(path, 'r')
            context = file.readlines()
            for idx, val in enumerate(context):
                if idx > 0:
                    self.h_mass.append(float(val.strip()))
            file.close()

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

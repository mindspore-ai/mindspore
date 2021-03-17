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
"""Langevin Liujian MD class"""

import math

import numpy as np
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype


class Langevin_Liujian:
    """Langevin_Liujian class"""

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
        self.rand_seed = 0 if "langevin_seed" not in controller.Command_Set else float(
            controller.Command_Set["langevin_seed"])  # jiahong0315
        self.max_velocity = 10000.0 if "velocity_max" not in controller.Command_Set else float(
            controller.Command_Set["velocity_max"])
        assert self.max_velocity > 0
        self.is_max_velocity = 0 if "velocity_max" not in controller.Command_Set else 1
        print("target temperature is ", self.target_temperature)
        print("friction coefficient is ", self.gamma_ln, "ps^-1")
        print("random seed is ", self.rand_seed)
        self.dt = float(controller.Command_Set["dt"])
        self.dt *= self.CONSTANT_TIME_CONVERTION
        self.half_dt = 0.5 * self.dt
        self.float4_numbers = math.ceil(3.0 * self.atom_numbers / 4.0)
        self.gamma_ln = self.gamma_ln / self.CONSTANT_TIME_CONVERTION
        self.exp_gamma = math.exp(-1 * self.gamma_ln * self.dt)
        self.sqrt_gamma = math.sqrt((1. - self.exp_gamma * self.exp_gamma) * self.target_temperature * self.CONSTANT_kB)
        self.h_sqrt_mass = [0] * self.atom_numbers
        for i in range(self.atom_numbers):
            self.h_sqrt_mass[i] = self.sqrt_gamma * math.sqrt(1. / self.h_mass[i])
        self.d_sqrt_mass = Tensor(self.h_sqrt_mass, mstype.float32)

    def read_information_from_amberfile(self, file_path):
        """read information from amberfile"""
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

    def MDIterationLeapFrog_Liujian(self, atom_numbers, half_dt, dt, exp_gamma, inverse_mass, sqrt_mass_inverse, vel,
                                    crd, frc, random_frc):
        """compute MDIterationLeapFrog Liujian"""
        inverse_mass = inverse_mass.reshape((-1, 1))
        sqrt_mass_inverse = sqrt_mass_inverse.reshape((-1, 1))
        acc = inverse_mass * frc
        vel = vel + dt * acc
        crd = crd + half_dt * vel
        vel = exp_gamma * vel + sqrt_mass_inverse * random_frc
        crd = crd + half_dt * vel
        frc = Tensor(np.zeros((atom_numbers, 3)), mstype.float32)
        return vel, crd, frc, acc

    def MD_Iteration_Leap_Frog(self, d_mass_inverse, vel_in, crd_in, frc_in):
        """MD_Iteration_Leap_Frog"""
        np.random.seed(int(self.rand_seed))
        self.rand_force = Tensor(np.zeros((self.atom_numbers, 3)), mstype.float32)
        # self.rand_force = Tensor(np.random.randn(self.atom_numbers, 3), mstype.float32)
        vel, crd, frc, acc = self.MDIterationLeapFrog_Liujian(atom_numbers=self.atom_numbers, half_dt=self.half_dt,
                                                              dt=self.dt, exp_gamma=self.exp_gamma,
                                                              inverse_mass=d_mass_inverse,
                                                              sqrt_mass_inverse=self.d_sqrt_mass,
                                                              vel=vel_in, crd=crd_in,
                                                              frc=frc_in, random_frc=self.rand_force)
        return vel, crd, frc, acc

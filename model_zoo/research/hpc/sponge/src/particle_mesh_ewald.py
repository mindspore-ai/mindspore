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
"""PME"""

import math
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, nn
from mindspore.ops import operations as P


class Particle_Mesh_Ewald(nn.Cell):
    """class Particle_Mesh_Ewald"""

    def __init__(self, controller, md_info):
        super(Particle_Mesh_Ewald, self).__init__()
        self.cutoff = 10.0 if "cut" not in controller.Command_Set else float(controller.Command_Set["cut"])
        self.tolerance = 0.00001 if "PME_Direct_Tolerance" not in controller.Command_Set else float(
            controller.Command_Set["PME_Direct_Tolerance"])
        self.fftx = -1 if "fftx" not in controller.Command_Set else int(controller.Command_Set["fftx"])
        self.ffty = -1 if "ffty" not in controller.Command_Set else int(controller.Command_Set["ffty"])
        self.fftz = -1 if "fftz" not in controller.Command_Set else int(controller.Command_Set["fftz"])
        self.atom_numbers = md_info.atom_numbers
        self.box_length = md_info.box_length

        if self.fftx < 0:
            self.fftx = self.Get_Fft_Patameter(self.box_length[0])
        if self.ffty < 0:
            self.ffty = self.Get_Fft_Patameter(self.box_length[1])
        if self.fftz < 0:
            self.fftz = self.Get_Fft_Patameter(self.box_length[2])

        self.beta = self.Get_Beta(self.cutoff, self.tolerance)
        self.box_length = Tensor(np.asarray(self.box_length, np.float32), mstype.float32)

        print("========== ", self.fftx, self.ffty, self.fftz, self.tolerance, self.beta)

    def Get_Beta(self, cutoff, tolerance):
        """get beta"""
        high = 1.0
        ihigh = 1
        while 1:
            tempf = math.erfc(high * cutoff) / cutoff
            if tempf <= tolerance:
                break
            high *= 2
            ihigh += 1
        ihigh += 50
        low = 0.0
        for _ in range(1, ihigh):
            beta = (low + high) / 2
            tempf = math.erfc(beta * cutoff) / cutoff
            if tempf >= tolerance:
                low = beta
            else:
                high = beta
        return beta

    def Check_2357_Factor(self, number):
        """check 2357 factor"""
        while number > 0:
            if number == 1:
                return 1
            tempn = number / 2
            if tempn * 2 != number:
                break
            number = tempn
        while number > 0:
            if number == 1:
                return 1
            tempn = number / 3
            if tempn * 3 != number:
                break
            number = tempn
        while number > 0:
            if number == 1:
                return 1
            tempn = number / 5
            if tempn * 5 != number:
                break
            number = tempn
        while number > 0:
            if number == 1:
                return 1
            tempn = number / 7
            if tempn * 7 != number:
                break
            number = tempn
        return 0

    def Get_Fft_Patameter(self, length):
        """get fft parameter"""
        tempi = math.ceil(length + 3) >> 2 << 2
        if 60 <= tempi <= 68:
            tempi = 64
        elif 120 <= tempi <= 136:
            tempi = 128
        elif 240 <= tempi <= 272:
            tempi = 256
        elif 480 <= tempi <= 544:
            tempi = 512
        elif 960 <= tempi <= 1088:
            tempi = 1024
        while 1:
            if self.Check_2357_Factor(tempi):
                return tempi
            tempi += 4

    def PME_Energy(self, uint_crd, charge, nl_atom_numbers, nl_atom_serial, uint_dr_to_dr_cof, excluded_list_start,
                   excluded_list, excluded_numbers, excluded_atom_numbers):
        """PME_Energy"""
        self.pmee = P.PMEEnergy(self.atom_numbers, excluded_atom_numbers, self.beta, self.fftx, self.ffty, self.fftz)
        self.reciprocal_energy, self.self_energy, self.direct_energy, self.correction_energy = \
            self.pmee(self.box_length, uint_crd, charge, nl_atom_numbers, nl_atom_serial, uint_dr_to_dr_cof,
                      excluded_list_start, excluded_list, excluded_numbers)
        return self.reciprocal_energy, self.self_energy, self.direct_energy, self.correction_energy

    def PME_Excluded_Force(self, uint_crd, scaler, charge, excluded_list_start, excluded_list,
                           excluded_numbers, excluded_atom_numbers):
        """PME Excluded Force"""
        self.pmeef = P.PMEExcludedForce(atom_numbers=self.atom_numbers, excluded_numbers=excluded_atom_numbers,
                                        beta=self.beta)
        self.frc = self.pmeef(uint_crd, scaler, charge, excluded_list_start, excluded_list, excluded_numbers)
        return self.frc

    def PME_Reciprocal_Force(self, uint_crd, charge):
        """PME reciprocal force"""
        self.pmerf = P.PMEReciprocalForce(self.atom_numbers, self.beta, self.fftx, self.ffty, self.fftz)
        self.frc = self.pmerf(self.box_length, uint_crd, charge)
        return self.frc

    def Energy_Device_To_Host(self):
        """Energy_Device_To_Host"""
        self.ee_ene = self.reciprocal_energy + self.self_energy + self.direct_energy + self.correction_energy
        return self.ee_ene

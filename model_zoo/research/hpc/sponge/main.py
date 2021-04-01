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
'''main'''
import argparse
import time

from src.simulation import Simulation

import mindspore.context as context
from mindspore import Tensor

parser = argparse.ArgumentParser(description='Sponge Controller')
parser.add_argument('--i', type=str, default=None, help='input file')
parser.add_argument('--amber_parm', type=str, default=None, help='paramter file in AMBER type')
parser.add_argument('--c', type=str, default=None, help='initial coordinates file')
parser.add_argument('--r', type=str, default="restrt", help='')
parser.add_argument('--x', type=str, default="mdcrd", help='')
parser.add_argument('--o', type=str, default="mdout", help="")
parser.add_argument('--box', type=str, default="mdbox", help='')
parser.add_argument('--device_id', type=int, default=0, help='')
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_opt.device_id, save_graphs=False)

if __name__ == "__main__":
    simulation = Simulation(args_opt)
    start = time.time()
    compiler_time = 0
    save_path = args_opt.o
    file = open(save_path, 'w')
    for steps in range(simulation.md_info.step_limit):
        print_step = steps % simulation.ntwx
        if steps == simulation.md_info.step_limit - 1:
            print_step = 0
        temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene, sigma_of_dihedral_ene, \
        nb14_lj_energy_sum, nb14_cf_energy_sum, LJ_energy_sum, ee_ene, _ = simulation(Tensor(steps), Tensor(print_step))
        if steps == 0:
            compiler_time = time.time()
        if steps % simulation.ntwx == 0 or steps == simulation.md_info.step_limit - 1:
            if steps == 0:
                print("_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ "
                      "_ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_")
                file.write("_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ "
                           "_ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_\n")

            temperature = temperature.asnumpy()
            total_potential_energy = total_potential_energy.asnumpy()
            print("{:>7.0f} {:>7.3f} {:>11.3f}".format(steps, float(temperature), float(total_potential_energy)),
                  end=" ")
            if simulation.bond.bond_numbers > 0:
                sigma_of_bond_ene = sigma_of_bond_ene.asnumpy()
                print("{:>10.3f}".format(float(sigma_of_bond_ene)), end=" ")
            if simulation.angle.angle_numbers > 0:
                sigma_of_angle_ene = sigma_of_angle_ene.asnumpy()
                print("{:>11.3f}".format(float(sigma_of_angle_ene)), end=" ")
            if simulation.dihedral.dihedral_numbers > 0:
                sigma_of_dihedral_ene = sigma_of_dihedral_ene.asnumpy()
                print("{:>14.3f}".format(float(sigma_of_dihedral_ene)), end=" ")
            if simulation.nb14.nb14_numbers > 0:
                nb14_lj_energy_sum = nb14_lj_energy_sum.asnumpy()
                nb14_cf_energy_sum = nb14_cf_energy_sum.asnumpy()
                print("{:>10.3f} {:>10.3f}".format(float(nb14_lj_energy_sum), float(nb14_cf_energy_sum)), end=" ")
            LJ_energy_sum = LJ_energy_sum.asnumpy()
            ee_ene = ee_ene.asnumpy()
            print("{:>7.3f}".format(float(LJ_energy_sum)), end=" ")
            print("{:>12.3f}".format(float(ee_ene)))
            if file is not None:
                file.write("{:>7.0f} {:>7.3f} {:>11.3f} {:>10.3f} {:>11.3f} {:>14.3f} {:>10.3f} {:>10.3f} {:>7.3f}"
                           " {:>12.3f}\n".format(steps, float(temperature), float(total_potential_energy),
                                                 float(sigma_of_bond_ene), float(sigma_of_angle_ene),
                                                 float(sigma_of_dihedral_ene), float(nb14_lj_energy_sum),
                                                 float(nb14_cf_energy_sum), float(LJ_energy_sum), float(ee_ene)))

    end = time.time()
    file.close()
    print("Main time(s):", end - start)
    print("Main time(s) without compiler:", end - compiler_time)

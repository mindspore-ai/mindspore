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

import mindspore.context as context
from mindspore import Tensor
from mindspore import load_checkpoint
from src.mdnn import Mdnn, TransCrdToCV
from src.simulation import Simulation

parser = argparse.ArgumentParser(description='SPONGE Controller')
parser.add_argument('--i', type=str, default=None, help='Input .in file')
parser.add_argument('--amber_parm', type=str, default=None, help='Paramter file in AMBER type')
parser.add_argument('--c', type=str, default=None, help='Initial coordinates file')
parser.add_argument('--r', type=str, default="restrt", help='')
parser.add_argument('--x', type=str, default="mdcrd", help='')
parser.add_argument('--o', type=str, default="mdout", help='Output file')
parser.add_argument('--box', type=str, default="mdbox", help='')
parser.add_argument('--device_id', type=int, default=0, help='GPU device id')
parser.add_argument('--u', type=bool, default=False, help='If use mdnn to update the atom charge')
parser.add_argument('--checkpoint', type=str, default="", help='Checkpoint file')
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_opt.device_id, save_graphs=False)
# context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=args_opt.device_id, save_graphs=False)

if __name__ == "__main__":
    simulation = Simulation(args_opt)
    if args_opt.u and args_opt.checkpoint:
        net = Mdnn()
        load_checkpoint(args_opt.checkpoint, net=net)
        transcrd = TransCrdToCV(simulation)

    start = time.time()
    compiler_time = 0
    save_path = args_opt.o
    simulation.Main_Initial()
    for steps in range(simulation.md_info.step_limit):
        print_step = steps % simulation.ntwx
        if steps == simulation.md_info.step_limit - 1:
            print_step = 0
        temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene, sigma_of_dihedral_ene, \
        nb14_lj_energy_sum, nb14_cf_energy_sum, LJ_energy_sum, ee_ene, _, _, _, _ = simulation(Tensor(steps),
                                                                                               Tensor(print_step))

        if steps == 0:
            compiler_time = time.time()
        if steps % simulation.ntwx == 0 or steps == simulation.md_info.step_limit - 1:
            simulation.Main_Print(steps, temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene,
                                  sigma_of_dihedral_ene, nb14_lj_energy_sum, nb14_cf_energy_sum, LJ_energy_sum, ee_ene)

        if args_opt.u and args_opt.checkpoint and steps % (4 * simulation.ntwx) == 0:
            print("Update charge!")
            inputs = transcrd(Tensor(simulation.crd), Tensor(simulation.last_crd))
            t_charge = net(inputs)
            simulation.charge = transcrd.updatecharge(t_charge)

    end = time.time()
    print("Main time(s):", end - start)
    print("Main time(s) without compiler:", end - compiler_time)
    simulation.Main_Destroy()

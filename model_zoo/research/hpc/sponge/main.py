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
"""main"""

import time
import argparse
from mindspore import context
from src.simulation_initial import Simulation

parser = argparse.ArgumentParser(description='Sponge Controller')
parser.add_argument('--i', type=str, default=None, help='input file')
parser.add_argument('--amber_parm', type=str, default=None,
                    help='paramter file in AMBER type')
parser.add_argument('--c', type=str, default=None,
                    help='initial coordinates file')
parser.add_argument('--r', type=str, default="restrt", help='')
parser.add_argument('--x', type=str, default="mdcrd", help='')
parser.add_argument('--o', type=str, default="mdout", help="")
parser.add_argument('--box', type=str, default="mdbox", help='')
args_opt = parser.parse_args()

context.set_context(mode=context.PYNATIVE_MODE,
                    device_target="GPU", device_id=0, save_graphs=True)

if __name__ == "__main__":
    start = time.time()
    simulation = Simulation(args_opt)
    simulation.Main_Initial()
    res = simulation.Initial_Neighbor_List_Update(not_first_time=0)
    md_info = simulation.md_info
    md_info.step_limit = 1
    for i in range(1, md_info.step_limit + 1):
        print("steps: ", i)
        md_info.steps = i
        simulation.Main_Before_Calculate_Force()
        simulation.Main_Calculate_Force()
        simulation.Main_Calculate_Energy()
        simulation.Main_After_Calculate_Energy()
        temperature = simulation.Main_Print()
        simulation.Main_Iteration_2()
    end = time.time()
    print("Main time(s):", end - start)
    simulation.Main_Destroy()

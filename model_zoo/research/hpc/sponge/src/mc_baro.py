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
'''MC_BARO'''


class MC_BARO:
    '''MC_BARO'''

    def __init__(self, controller, atom_numbers, target_pressure, boxlength, res_is_initialized, mode):
        self.constant_pres_convertion = 6.946827162543585e4
        print("START INITIALIZING MC BAROSTAT:\n")
        self.module_name = "mc_baro"
        self.atom_numbers = atom_numbers
        self.accept_rate = 0
        self.DeltaV_max = 200.0
        self.is_controller_printf_initialized = 0
        self.is_initialized = 0
        # initial
        self.target_pressure = target_pressure
        print("    The target pressure is %.2f bar\n" % (target_pressure * self.constant_pres_convertion))
        self.V0 = boxlength[0] * boxlength[1] * boxlength[2]
        self.newV = self.V0
        self.mc_baro_initial_ratio = 0.01 if "initial_ratio" not in controller.Command_Set else float(
            controller.Command_Set["initial_ratio"])
        self.DeltaV_max = self.mc_baro_initial_ratio * self.V0
        print("    The initial max volume to change is %f A^3\n" % (self.DeltaV_max))
        self.update_interval = 100 if "update_interval" not in controller.Command_Set else int(
            controller.Command_Set["update_interval"])
        print("    The update_interval is %d\n" % (self.update_interval))
        self.check_interval = 20 if "check_interval" not in controller.Command_Set else int(
            controller.Command_Set["check_interval"])
        print("    The check_interval is %d\n" % (self.check_interval))
        self.scale_coordinate_by_residue = res_is_initialized if "residue_scale" not in controller.Command_Set else int(
            controller.Command_Set["residue_scale"])
        if self.scale_coordinate_by_residue == 1 and res_is_initialized == 0:
            print(
                "    Warning: The residue is not initialized. Atom scale mode is set instead.\n")
            self.scale_coordinate_by_residue = 0
        print("    The residue_scale is %d\n" % (self.scale_coordinate_by_residue))

        self.system_reinitializing_count = 0
        self.accept_rate_low = 30.0 if "accept_rate_low" not in controller.Command_Set else int(
            controller.Command_Set["accept_rate_low"])
        print("    The lowest accept rate is %.2f\n" % (self.accept_rate_low))
        self.accept_rate_high = 40.0 if "accept_rate_high" not in controller.Command_Set else int(
            controller.Command_Set["accept_rate_high"])
        print("    The highest accept rate is %.2f\n" % (self.accept_rate_high))
        self.frc_backup = 0  # [atom_numbers, 3]
        self.crd_backup = 0  # [atom_numbers, 3]
        if mode == 2 and controller.Command_Set["barostat"] == "monte_carlo":
            self.is_initialized = 1
        else:
            self.is_initialized = 0
        if self.is_initialized and not self.is_controller_printf_initialized:
            self.is_controller_printf_initialized = 1
        print("END INITIALIZING MC BAROSTAT\n")

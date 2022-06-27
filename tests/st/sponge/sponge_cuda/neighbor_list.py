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
'''Neighbor List'''


class NeighborList:
    '''Neighbor List'''

    def __init__(self, controller, atom_numbers, box_length):
        self.constant_unit_max_float = 4294967296.0
        print("START INITIALIZING NEIGHBOR LIST:")
        self.module_name = "NeighborList"
        self.refresh_interval = 20 if "refresh_interval" not in controller.command_set else int(
            controller.command_set["refresh_interval"])
        self.max_atom_in_grid_numbers = 64 if "max_atom_in_grid_numbers" not in controller.command_set else int(
            controller.command_set["max_atom_in_grid_numbers"])
        self.max_neighbor_numbers = 800 if "max_neighbor_numbers" not in controller.command_set else int(
            controller.command_set["max_neighbor_numbers"])

        self.skin = 2.0 if "skin" not in controller.command_set else float(controller.command_set["skin"])
        self.cutoff = 10.0 if "cutoff" not in controller.command_set else float(controller.command_set["cutoff"])
        self.cutoff_square = self.cutoff * self.cutoff
        self.cutoff_with_skin = self.cutoff + self.skin
        self.half_cutoff_with_skin = 0.5 * self.cutoff_with_skin
        self.cutoff_with_skin_square = self.cutoff_with_skin * self.cutoff_with_skin
        self.half_skin_square = 0.25 * self.skin * self.skin
        self.atom_numbers = atom_numbers
        self.box_length = box_length
        self.update_volume()

        self.initial_neighbor_grid()
        self.not_first_time = 0
        self.is_initialized = 1
        self.refresh_count = [0]

        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)

    def read_information_from_amberfile(self, file_path):
        '''read amber file'''
        file = open(file_path, 'r')
        context = file.readlines()
        file.close()
        self.excluded_list_start = [0] * self.atom_numbers
        self.excluded_numbers = [0] * self.atom_numbers

        for idx, val in enumerate(context):
            if idx < len(context) - 1:
                if "%FLAG POINTERS" in val + context[idx + 1] and "%FORMAT(10I8)" in val + context[idx + 1]:
                    start_idx = idx + 2
                    count = 0
                    value = list(map(int, context[start_idx].strip().split()))
                    information = []
                    information.extend(value)
                    while count < 11:
                        start_idx += 1
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                    self.excluded_atom_numbers = information[10]
                    print("excluded atom numbers ", self.excluded_atom_numbers)
                    break
        for idx, val in enumerate(context):
            if "%FLAG NUMBER_EXCLUDED_ATOMS" in val:
                count = 0
                start_idx = idx
                information = []
                while count < self.atom_numbers:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)
                count = 0
                for i in range(self.atom_numbers):
                    self.excluded_numbers[i] = information[i]
                    self.excluded_list_start[i] = count
                    count += information[i]
                break

        total_count = sum(self.excluded_numbers)
        self.excluded_list = []
        for idx, val in enumerate(context):
            if "%FLAG EXCLUDED_ATOMS_LIST" in val:
                count = 0
                start_idx = idx
                information = []
                while count < total_count:
                    start_idx += 1
                    if "%FORMAT" in context[start_idx]:
                        continue
                    else:
                        value = list(map(int, context[start_idx].strip().split()))
                        information.extend(value)
                        count += len(value)

                count = 0
                for i in range(self.atom_numbers):
                    tmp_list = []
                    if self.excluded_numbers[i] == 1:
                        tmp_list.append(information[count] - 1)
                        if information[count] == 0:
                            self.excluded_numbers[i] = 0
                        count += 1
                    else:
                        for _ in range(self.excluded_numbers[i]):
                            tmp_list.append(information[count] - 1)

                            count += 1
                        tmp_list = sorted(tmp_list)
                    self.excluded_list.extend(tmp_list)
                break

    def initial_neighbor_grid(self):
        '''init neighbor grid'''
        half_cutoff = self.half_cutoff_with_skin
        self.nx = int(self.box_length[0] / half_cutoff)
        self.ny = int(self.box_length[1] / half_cutoff)
        self.nz = int(self.box_length[2] / half_cutoff)
        self.grid_n = [self.nx, self.ny, self.nz]
        self.grid_length = [self.box_length[0] / self.nx,
                            self.box_length[1] / self.ny,
                            self.box_length[2] / self.nz]
        self.grid_length_inverse = [1.0 / self.grid_length[0], 1.0 / self.grid_length[1], 1.0 / self.grid_length[2]]

        self.nxy = self.nx * self.ny
        self.grid_numbers = self.nz * self.nxy
        self.atom_numbers_in_grid_bucket = [0] * self.grid_numbers
        self.bucket = [-1] * (self.grid_numbers * self.max_atom_in_grid_numbers)

        self.pointer = []
        temp_grid_serial = [0] * 125
        for i in range(self.grid_numbers):
            nz = int(i / self.nxy)
            ny = int((i - self.nxy * nz) / self.nx)
            nx = i - self.nxy * nz - self.nx * ny
            count = 0
            for l in range(-2, 3):
                for m in range(-2, 3):
                    for n in range(-2, 3):
                        xx = nx + l
                        if xx < 0:
                            xx = xx + self.nx
                        elif xx >= self.nx:
                            xx = xx - self.nx
                        yy = ny + m
                        if yy < 0:
                            yy = yy + self.ny
                        elif yy >= self.ny:
                            yy = yy - self.ny
                        zz = nz + n
                        if zz < 0:
                            zz = zz + self.nz
                        elif zz >= self.nz:
                            zz = zz - self.nz
                        temp_grid_serial[count] = zz * self.nxy + yy * self.nx + xx
                        count += 1
            temp_grid_serial = sorted(temp_grid_serial)
            self.pointer.extend(temp_grid_serial)

    def update_volume(self):
        self.quarter_crd_to_uint_crd_cof = [0.25 * self.constant_unit_max_float / self.box_length[0],
                                            0.25 * self.constant_unit_max_float / self.box_length[1],
                                            0.25 * self.constant_unit_max_float / self.box_length[2]]
        self.uint_dr_to_dr_cof = [1.0 / self.constant_unit_max_float * self.box_length[0],
                                  1.0 / self.constant_unit_max_float * self.box_length[1],
                                  1.0 / self.constant_unit_max_float * self.box_length[2]]

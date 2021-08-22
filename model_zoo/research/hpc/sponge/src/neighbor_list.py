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
class neighbor_list:
    '''Neighbor List'''
    def __init__(self, controller, atom_numbers, box_length):
        self.refresh_interval = 20 if "neighbor_list_refresh_interval" not in controller.Command_Set else int(
            controller.Command_Set["neighbor_list_refresh_interval"])
        self.max_atom_in_grid_numbers = 64 if "max_atom_in_grid_numbers" not in controller.Command_Set else int(
            controller.Command_Set["max_atom_in_grid_numbers"])
        self.max_neighbor_numbers = 800 if "max_neighbor_numbers" not in controller.Command_Set else int(
            controller.Command_Set["max_neighbor_numbers"])
        self.skin = 2.0 if "skin" not in controller.Command_Set else float(controller.Command_Set["skin"])
        self.cutoff = 10.0 if "cut" not in controller.Command_Set else float(controller.Command_Set["cut"])
        self.cutoff_square = self.cutoff * self.cutoff
        self.cutoff_with_skin = self.cutoff + self.skin
        self.half_cutoff_with_skin = 0.5 * self.cutoff_with_skin
        self.cutoff_with_skin_square = self.cutoff_with_skin * self.cutoff_with_skin
        self.half_skin_square = 0.25 * self.skin * self.skin
        self.atom_numbers = atom_numbers
        self.box_length = box_length

        if controller.amber_parm is not None:
            file_path = controller.amber_parm
            self.read_information_from_amberfile(file_path)

        self.Initial_Neighbor_Grid()
        self.not_first_time = 0
        self.refresh_count = [0]

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

    def Initial_Neighbor_Grid(self):
        '''init neighbor grid'''
        half_cutoff = self.half_cutoff_with_skin
        self.Nx = int(self.box_length[0] / half_cutoff)
        self.Ny = int(self.box_length[1] / half_cutoff)
        self.Nz = int(self.box_length[2] / half_cutoff)
        self.grid_N = [self.Nx, self.Ny, self.Nz]
        self.grid_length = [self.box_length[0] / self.Nx, self.box_length[1] / self.Ny, self.box_length[2] / self.Nz]
        self.grid_length_inverse = [1.0 / self.grid_length[0], 1.0 / self.grid_length[1], 1.0 / self.grid_length[2]]
        self.Nxy = self.Nx * self.Ny
        self.grid_numbers = self.Nz * self.Nxy

        self.atom_numbers_in_grid_bucket = [0] * self.grid_numbers
        self.bucket = [-1] * (self.grid_numbers * self.max_atom_in_grid_numbers)
        self.pointer = []
        temp_grid_serial = [0] * 125
        for i in range(self.grid_numbers):
            Nz = int(i / self.Nxy)
            Ny = int((i - self.Nxy * Nz) / self.Nx)
            Nx = i - self.Nxy * Nz - self.Nx * Ny
            count = 0
            for l in range(-2, 3):
                for m in range(-2, 3):
                    for n in range(-2, 3):
                        xx = Nx + l
                        if xx < 0:
                            xx = xx + self.Nx
                        elif xx >= self.Nx:
                            xx = xx - self.Nx
                        yy = Ny + m
                        if yy < 0:
                            yy = yy + self.Ny
                        elif yy >= self.Ny:
                            yy = yy - self.Ny
                        zz = Nz + n
                        if zz < 0:
                            zz = zz + self.Nz
                        elif zz >= self.Nz:
                            zz = zz - self.Nz
                        temp_grid_serial[count] = zz * self.Nxy + yy * self.Nx + xx
                        count += 1
            temp_grid_serial = sorted(temp_grid_serial)
            self.pointer.extend(temp_grid_serial)

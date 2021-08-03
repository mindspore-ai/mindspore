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
'''Coordinate Molecular Map'''
import collections
import numpy as np


class CoordinateMolecularMap:
    '''Coordinate Molecular Map'''

    def __init__(self, atom_numbers, box_length, crd, exclude_numbers, exclude_length, exclude_start, exclude_list):
        self.module_name = "crd_mole_wrap"
        self.atom_numbers = atom_numbers
        self.box_length = box_length
        self.crd = crd
        # self.coordinate = crd
        self.exclude_numbers = exclude_numbers
        self.exclude_length = exclude_length
        self.exclude_start = exclude_start
        self.exclude_list = exclude_list
        print("START INITIALIZING Coordinate Molecular Map:\n")
        self.h_box_map_times = np.zeros([self.atom_numbers, 3])
        self.h_old_crd = self.crd
        self.h_nowrap_crd = self.crd
        self.move_crd_nearest_from_exclusions_host()
        self.is_initialized = 1
        print("END INITIALIZING Coordinate Molecular Map\n")

    def move_crd_nearest_from_exclusions_host(self):
        '''move_crd_nearest_from_exclusions_host'''
        edge_numbers = 2 * self.exclude_numbers
        visited = [0] * self.atom_numbers
        first_edge = [-1] * self.atom_numbers
        edges = [0] * edge_numbers
        edge_next = [0] * edge_numbers
        atom_i, atom_j, edge_count = 0, 0, 0
        for i in range(self.atom_numbers):
            atom_i = i
            for j in range(self.exclude_start[i] + self.exclude_length[i] - 1, self.exclude_start[i] - 1, -1):
                atom_j = self.exclude_list[j]
                edge_next[edge_count] = first_edge[atom_i]
                first_edge[atom_i] = edge_count
                edges[edge_count] = atom_j
                edge_count += 1
                edge_next[edge_count] = first_edge[atom_j]
                first_edge[atom_j] = edge_count
                edges[edge_count] = atom_i
                edge_count += 1
        queue = collections.deque()

        for i in range(self.atom_numbers):
            if not visited[i]:
                visited[i] = 1
                queue.append(i)
                atom_front = i
                while queue:
                    atom = queue[0]
                    queue.popleft()
                    self.h_box_map_times[atom][0] = int(
                        (self.crd[atom_front][0] - self.crd[atom][0]) / self.box_length[0] \
                        + 0.5)
                    self.h_box_map_times[atom][1] = int(
                        (self.crd[atom_front][1] - self.crd[atom][1]) / self.box_length[1] \
                        + 0.5)
                    self.h_box_map_times[atom][2] = int(
                        (self.crd[atom_front][2] - self.crd[atom][2]) / self.box_length[1] \
                        + 0.5)
                    self.crd[atom][0] = self.crd[atom][0] + self.h_box_map_times[atom][0] * self.box_length[0]
                    self.crd[atom][1] = self.crd[atom][1] + self.h_box_map_times[atom][1] * self.box_length[1]
                    self.crd[atom][2] = self.crd[atom][2] + self.h_box_map_times[atom][2] * self.box_length[2]
                    edge_count = first_edge[atom]
                    atom_front = atom
                    while edge_count is not -1:
                        atom = edges[edge_count]
                        if not visited[atom]:
                            queue.append(atom)
                            visited[atom] = 1
                        edge_count = edge_next[edge_count]

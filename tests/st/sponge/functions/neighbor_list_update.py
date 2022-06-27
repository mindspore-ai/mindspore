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
'''neighbor list update'''
from mindspore import numpy as np
from mindspore import ops, Tensor
from .common import get_periodic_displacement, get_range_tensor, get_neighbour_index
from .crd_to_uint_crd import crd_to_uint_crd

step = Tensor(1, np.int32)


def not_excluded_mask(atom_numbers, excluded_list_start, excluded_list, excluded_numbers):
    not_excluded = np.full((atom_numbers, atom_numbers), True, np.bool_)
    for i, v in enumerate(excluded_list_start):
        if excluded_numbers[i] > 0:
            excluded_serial = excluded_list[ops.tensor_range(v, v + excluded_numbers[i], step)]
            not_excluded[i, excluded_serial] = False
    return not_excluded


def find_atom_neighbors(atom_numbers, uint_crd, uint_dr_to_dr_cof, cutoff_skin_square):
    dr = get_periodic_displacement(uint_crd, np.expand_dims(uint_crd, -2), uint_dr_to_dr_cof)
    dr2 = np.sum(dr ** 2, -1)
    atom_idx = get_range_tensor(atom_numbers)
    nl_mask = np.logical_and(atom_idx.reshape(-1, 1) < atom_idx, dr2 < cutoff_skin_square)
    return nl_mask


def delete_excluded_atoms_serial_in_neighbor_list(
        atom_numbers, max_neighbor_numbers, nl_mask, not_excluded):
    mask = np.logical_and(nl_mask, not_excluded)
    serial_idx = get_neighbour_index(atom_numbers, atom_numbers)
    nl_serial = np.where(mask, serial_idx, atom_numbers)
    nl_serial = np.sort(nl_serial, -1)[:, : max_neighbor_numbers]
    nl_numbers = np.sum(mask, -1)
    return nl_numbers, nl_serial


def crd_periodic_map(crd, box_length):
    crd = np.where(crd < 0, crd + box_length, crd)
    crd = np.where(crd > box_length, crd - box_length, crd)
    return crd


def find_atom_in_grid_serial(grid_length_inverse, crd, grid_n, nxy, atom_in_grid_serial):
    grid_idx = (crd * grid_length_inverse).astype(np.int32)
    grid_idx = np.where(grid_idx < grid_n, grid_idx, 0)
    atom_in_grid_serial = grid_idx[..., 2] * nxy + grid_idx[..., 1] * grid_n[0] + grid_idx[..., 0]
    return atom_in_grid_serial


def neighbor_list_update(
        grid_numbers, atom_numbers, not_first_time, nxy, excluded_atom_numbers,
        cutoff_square, half_skin_square, cutoff_with_skin, half_cutoff_with_skin, cutoff_with_skin_square,
        refresh_interval, cutoff, skin, max_atom_in_grid_numbers, max_neighbor_numbers,
        atom_numbers_in_grid_bucket, bucket, crd, box_length, grid_n, grid_length_inverse, atom_in_grid_serial,
        old_crd, crd_to_uint_crd_cof, uint_crd, gpointer, nl_atom_numbers, nl_atom_serial, uint_dr_to_dr_cof,
        not_excluded, need_refresh_flag, refresh_count):
    """
    Update (or construct if first time) the Verlet neighbor list for the
    calculation of short-ranged force. Assume the number of atoms is n,
    the number of grids divided is G, the maximum number of atoms in one
    grid is m, the maximum number of atoms in single atom's neighbor list
    is L, and the number of total atom in excluded list is E.

    Args:
        grid_numbers (int32): the total number of grids divided.
        not_first_time (int32): whether to construct the neighbor
            list first time or not.
        nxy (int32): the total number of grids divided in xy plane.
        excluded_atom_numbers (int32): the total atom numbers in the excluded list.
        cutoff (float32): the cutoff distance for short-range force calculation. Default: 10.0.
        skin (float32): the overflow value of cutoff to maintain a neighbor list. Default: 2.0.
        cutoff_square (float32): the suqare value of cutoff.
        half_skin_square (float32): skin*skin/4, indicates the maximum
            square value of the distance atom allowed to move between two updates.
        cutoff_with_skin (float32): cutoff + skin, indicates the
            radius of the neighbor list for each atom.
        half_cutoff_with_skin (float32): cutoff_with_skin/2.
        cutoff_with_skin_square (float32): the square value of cutoff_with_skin.
        refresh_interval (int32): the number of iteration steps between two updates of neighbor
            list. Default: 20.
        max_atom_in_grid_numbers (int32): the maximum number of atoms in one grid. Default: 64.
        max_neighbor_numbers (int32): The maximum number of neighbors. Default: 800.
        atom_numbers_in_grid_bucket (Tensor, int32) - [G,], the number of atoms in each grid bucket.
        bucket (Tensor, int32) - (Tensor,int32) - [G, m], the atom indices in each grid bucket.
        crd (Tensor, float32) - [n,], the coordinates of each atom.
        box_length (Tensor, float32) - [3,], the length of 3 dimensions of the simulation box.
        grid_n (Tensor, int32) - [3,], the number of grids divided of 3 dimensions of the
            simulation box.
        grid_length_inverse (float32) - the inverse value of grid length.
        atom_in_grid_serial (Tensor, int32) - [n,], the grid index for each atom.
        old_crd (Tensor, float32) - [n, 3], the coordinates before update of each atom.
        crd_to_uint_crd_cof (Tensor, float32) - [3,], the scale factor
            between the unsigned int value and the real space coordinates.
        uint_crd (Tensor, uint32) - [n, 3], the unsigned int coordinates value fo each atom.
        gpointer (Tensor, int32) - [G, 125], the 125 nearest neighbor grids (including self) of each
            grid. G is the number of nearest neighbor grids.
        nl_atom_numbers (Tensor, int32) - [n,], the number of atoms in neighbor list of each atom.
        nl_atom_serial (Tensor, int32) - [n, L], the indices of atoms in neighbor list of each atom.
        uint_dr_to_dr_cof (Tensor, float32) - [3,], the scale factor between
            the real space coordinates and the unsigned int value.
        excluded_list_start (Tensor, int32) - [n,], the start excluded index in excluded list for
            each atom.
        excluded_numbers (Tensor, int32) - [n,], the number of atom excluded in excluded list for
            each atom.
        not_excluded (Tensor, bool) - [n, n], marking the excluded atoms for each atom, where each
            element ij indicates whether atom j is not excluded for atom i.
        need_refresh_flag (Tensor, int32) - [n,], whether the neighbor list of each atom need update
            or not.
        refresh_count (Tensor, int32) - [1,], count how many iteration steps have passed since last
            update.

    Outputs:
        nl_atom_numbers (Tensor, int32) - [n,], the number of atoms in neighbor list of each atom.
        nl_atom_serial (Tensor, int32) - [n, L], the indices of atoms in neighbor list of each atom.
        crd (Tensor, float32) - [n,], the coordinates of each atom.
        old_crd (Tensor, float32) - [n, 3], the coordinates before update of each atom.
        need_refresh_flag (Tensor, int32) - [n,], whether the neighbor list of each atom need update
            or not.
        refresh_count (Tensor, int32) - [1,], count how many iteration steps have passed since last
            update.

    Supported Platforms:
        ``GPU``
    """
    half_crd_to_uint_crd_cof = 0.5 * crd_to_uint_crd_cof
    if not_first_time:
        if refresh_interval > 0:
            refresh_cond = (refresh_count % refresh_interval) == 0
            trans_vec = np.full(3, -skin, np.float32)
            crd = np.where(refresh_cond, crd + trans_vec, crd)
            crd = np.where(refresh_cond, crd_periodic_map(crd, box_length), crd)
            crd = np.where(refresh_cond, crd - trans_vec, crd)
            old_crd = np.where(refresh_cond, crd, old_crd)

            uint_crd = np.where(refresh_cond,
                                crd_to_uint_crd(half_crd_to_uint_crd_cof, crd).astype(np.int32),
                                uint_crd.astype(np.int32)).astype(np.uint32)

            nl_mask = find_atom_neighbors(
                atom_numbers, uint_crd, uint_dr_to_dr_cof, cutoff_square)

            nl_atom_numbers_updated, nl_atom_serial_updated = delete_excluded_atoms_serial_in_neighbor_list(
                atom_numbers, max_neighbor_numbers, nl_mask, not_excluded)
            nl_atom_numbers = np.where(refresh_cond, nl_atom_numbers_updated, nl_atom_numbers)
            nl_atom_serial = np.where(refresh_cond, nl_atom_serial_updated, nl_atom_serial)

            refresh_count += 1
        else:
            r1 = crd - old_crd
            r1_2 = np.sum(r1, -1)
            if (r1_2 > half_skin_square).any():
                trans_vec = np.full(3, skin, np.float32)
                crd += trans_vec
                crd = crd_periodic_map(crd, box_length)
                crd -= trans_vec
                old_crd[...] = crd

                uint_crd = crd_to_uint_crd(half_crd_to_uint_crd_cof, crd)

                nl_mask = find_atom_neighbors(
                    atom_numbers, uint_crd, uint_dr_to_dr_cof, cutoff_with_skin_square)

                nl_atom_numbers, nl_atom_serial = delete_excluded_atoms_serial_in_neighbor_list(
                    atom_numbers, max_neighbor_numbers, nl_mask, not_excluded)

                need_refresh_flag[0] = 0
    else:
        trans_vec = np.full(3, skin, np.float32)
        crd = crd_periodic_map(crd, box_length)
        crd += trans_vec
        old_crd[...] = crd

        uint_crd = crd_to_uint_crd(half_crd_to_uint_crd_cof, crd)

        nl_mask = find_atom_neighbors(
            atom_numbers, uint_crd, uint_dr_to_dr_cof, cutoff_with_skin_square)

        nl_atom_numbers, nl_atom_serial = delete_excluded_atoms_serial_in_neighbor_list(
            atom_numbers, max_neighbor_numbers, nl_mask, not_excluded)

        res = (nl_atom_numbers, nl_atom_serial, crd, old_crd, need_refresh_flag, refresh_count)

    return res


# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""data transform MSA TEMPLATE"""
import numpy as np
from tests.st.mindscience.mindsponge.mindsponge.common.residue_constants import restype_1to3, chi_angles_atoms, \
    atom_order, restypes


def get_chi_atom_pos_indices():
    """get the atom indices for computing chi angles for all residue types"""
    chi_atom_pos_indices = []
    for residue_name in restypes:
        residue_name = restype_1to3.get(residue_name)
        residue_chi_angles = chi_angles_atoms.get(residue_name)
        atom_pos_indices = []
        for chi_angle in residue_chi_angles:
            atom_pos_indices.append([atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_pos_indices)):
            atom_pos_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_pos_indices.append(atom_pos_indices)

    chi_atom_pos_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return np.array(chi_atom_pos_indices)

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

"""Operators for sponge."""

from ..primitive import PrimitiveWithInfer, prim_attr_register
from ..._checkparam import Validator as validator
from ...common import dtype as mstype
from ..._checkparam import Rel


class BondForce(PrimitiveWithInfer):
    """
    BondForce:

    Calculate the force exerted by the simple harmonic bond on the
    corresponding atoms. Assume the number of harmonic bonds is M and
    the number of atoms is N.

    .. math::

        dr = (x_1-x_2, y_1-y_2, z_1-z_2)
        F = (F_x, F_y, F_z) = 2*k*(1 - r_0/|dr|)*dr

    Inputs:
        - **uint_crd_f** (Tensor, uint32 ) - [N, 3], the unsigned int coordinate
        value of each atom.
        - **scaler_f** (Tensor, float32) - [3, 1], the 3-D scale factor (x, y, z),
    between the real space float coordinates and the unsigned int coordinates.
        - **atom_a** (Tensor, int32) - [M, 1], the first atom index of each bond.
        - **atom_b** (Tensor, int32) - [M, 1], the second atom index of each bond.
        - **bond_k** (Tensor, float32) - [M, 1], the force constant of each bond.
        - **bond_r0** (Tensor, float32) - [M, 1], the equlibrium length of each bond.

    Outputs:
        - **frc_f** (float32 Tensor) - [N, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    Examples:
    """

    @prim_attr_register
    def __init__(self, bond_numbers):
        self.bond_numbers = bond_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'bond_k', 'bond_r0'],
                                outputs=['frc_f'])
        self.add_prim_attr('bond_numbers', self.bond_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, bond_k_shape, bond_r0_shape):
        cls_name = self.name
        # N = uint_crd_f_shape[0]
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            bond_k_shape[0], M, Rel.EQ, "bond_k_shape", cls_name)
        validator.check_int(
            bond_r0_shape[0], M, Rel.EQ, "bond_r0_shape", cls_name)
        return uint_crd_f_shape

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, bond_k_type, bond_r0_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('bond_k_type', bond_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('bond_r0_type', bond_r0_type, [mstype.float32], self.name)
        return bond_r0_type


class BondEnergy(PrimitiveWithInfer):
    """
    BondEnergyCuda:

    Calculate the harmonic potential energy between each bonded atom pair.
    Assume our system has N atoms and M harmonic bonds.

    .. math::

        dr = (x_1-x_2, y_1-y_2, z_1-z_2)
        E = k*(|dr| - r_0)^2

    Inputs:
        Same as operator BondForce().

    .. math::

        dr = (x_1-x_2, y_1-y_2, z_1-z_2)
        E = k*(|dr| - r_0)^2

    Outputs:
        - **bond_ene** (Tensor, float32) - [M, 1], the harmonic potential energy
        for each bond.

    Supported Platforms:
        ``GPU``
    Examples:
    """

    @prim_attr_register
    def __init__(self, bond_numbers):
        self.bond_numbers = bond_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'bond_k', 'bond_r0'],
                                outputs=['bond_ene'])
        self.add_prim_attr('bond_numbers', self.bond_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, bond_k_shape, bond_r0_shape):
        cls_name = self.name
        # N = uint_crd_f_shape[0]
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            bond_k_shape[0], M, Rel.EQ, "bond_k_shape", cls_name)
        validator.check_int(
            bond_r0_shape[0], M, Rel.EQ, "bond_r0_shape", cls_name)

        return bond_k_shape

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, bond_k_type, bond_r0_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('bond_k_type', bond_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('bond_r0_type', bond_r0_type, [mstype.float32], self.name)
        return bond_r0_type


class BondAtomEnergy(PrimitiveWithInfer):
    """
    BondAtomEnergyCuda:

    Add the potential energy caused by simple harmonic bonds to the total
    potential energy of each atom.

    The calculation formula is the same as operator BondEnergy().

    Inputs:
        Same as operator BondForce().

    Outputs:
        - **atom_ene** (Tensor, float32) - [N, 1], the accumulated potential
        energy for each atom.

    Supported Platforms:
        ``GPU``
    Examples:
    """

    @prim_attr_register
    def __init__(self, bond_numbers):
        self.bond_numbers = bond_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'bond_k', 'bond_r0'],
                                outputs=['atom_ene'])
        self.add_prim_attr('bond_numbers', self.bond_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, bond_k_shape, bond_r0_shape):
        cls_name = self.name
        N = uint_crd_f_shape[0]
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            bond_k_shape[0], M, Rel.EQ, "bond_k_shape", cls_name)
        validator.check_int(
            bond_r0_shape[0], M, Rel.EQ, "bond_r0_shape", cls_name)
        return [N,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, bond_k_type, bond_r0_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('bond_k_type', bond_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('bond_r0_type', bond_r0_type, [mstype.float32], self.name)
        return bond_r0_type


class BondForceWithAtomEnergy(PrimitiveWithInfer):
    """
    BondForceWithAtomEnergy:

    Calculate bond force and harmonic potential energy together.

    The calculation formula is the same as operator BondForce() and BondEnergy().

    Inputs:
        Same as operator BondForce().

    Outputs:
        - **frc_f** (Tensor, float32) - [N, 3], same as operator BondForce().
        - **atom_e** (Tensor, float32) - [N, 1], same as atom_ene in operator BondAtomEnergy().

    Supported Platforms:
        ``GPU``
    Examples:
    """

    @prim_attr_register
    def __init__(self, bond_numbers):
        self.bond_numbers = bond_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'bond_k', 'bond_r0'],
                                outputs=['frc_f', 'atom_e'])
        self.add_prim_attr('bond_numbers', self.bond_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, bond_k_shape, bond_r0_shape):
        cls_name = self.name
        N = uint_crd_f_shape[0]
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            bond_k_shape[0], M, Rel.EQ, "bond_k_shape", cls_name)
        validator.check_int(
            bond_r0_shape[0], M, Rel.EQ, "bond_r0_shape", cls_name)
        return uint_crd_f_shape, [N,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, bond_k_type, bond_r0_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('bond_k_type', bond_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('bond_r0_type', bond_r0_type, [mstype.float32], self.name)
        return bond_r0_type, bond_r0_type


class BondForceWithAtomVirial(PrimitiveWithInfer):
    """
    BondForceWithAtomVirial:

    Calculate bond force and the virial coefficient caused by simple harmonic
    bond for each atom together.

    The calculation formula of the force part is the same as operator BondForce().
    The Virial part is as follows:

    .. math::

        dr = (x_1-x_2, y_1-y_2, z_1-z_2)
        virial = |dr|*(|dr| - r_0)*k

    Inputs:
        Same as operator BondForce()

    Outputs:
        - **frc_f** (Tensor, float32) - [N, 3], same as operator BondForce().
        - **atom_v** (Tensor, float32) - [N, 1], the accumulated virial coefficient
        for each atom.

    Supported Platforms:
        ``GPU``
    Examples:
    """

    @prim_attr_register
    def __init__(self, bond_numbers):
        self.bond_numbers = bond_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'bond_k', 'bond_r0'],
                                outputs=['frc_f', 'atom_v'])
        self.add_prim_attr('bond_numbers', self.bond_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, bond_k_shape, bond_r0_shape):
        cls_name = self.name
        N = uint_crd_f_shape[0]
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            bond_k_shape[0], M, Rel.EQ, "bond_k_shape", cls_name)
        validator.check_int(
            bond_r0_shape[0], M, Rel.EQ, "bond_r0_shape", cls_name)
        return uint_crd_f_shape, [N,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, bond_k_type, bond_r0_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('bond_k_type', bond_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('bond_r0_type', bond_r0_type, [mstype.float32], self.name)
        return bond_r0_type, bond_r0_type


class DihedralForce(PrimitiveWithInfer):
    """
    DihedralForce:

    Calculate the force exerted by the dihedral term which made of 4-atoms
    on the corresponding atoms. Assume the number of dihedral terms is M and
    the number of atoms is N.

    .. math::

        dr_{ab} = (x_b-x_a, y_b-y_a, z_b-z_a)
        dr_{cb} = (x_b-x_c, y_b-y_c, z_b-z_c)
        dr_{cd} = (x_d-x_c, y_d-y_c, z_d-z_c)

        r1 = dr_{ab}*dr_{cb}
        r2 = dr_{cd}*dr_{cb}

        phi = pi - sign(inner_product(r1*r2), dr_{cb})
            * arccos(inner_product(r1, r2)/|r1|/|r2|)
        dEdphi = n*phi*(k*cos(phi_0)*sin(n*phi) - k*sin(phi_0)*cos(n*phi))/sin(phi)
        dphidr1 = r2/|r1|/|r2| + cos(phi)/|r1|^2*r1
        dphidr2 = r1/|r1|/|r2| + cos(phi)/|r2|^2*r2

        dEdra = dEdphi * dr_{cb} * dphidr1
        dEdrd = dEdphi * dphi_dr2 * dr_{cb}
        dEdrjpart = dEdphi * ((dr_{ab} * dphidr1) + (dr_{cd} * dphidr2))

        F_a = dEdri
        F_b = dEdrjpart - dEdri
        F_c = - dEdrl - dEdrjpart
        F_d = dEdrl

    Inputs:
        - **uint_crd_f** (Tensor, uint32) - [N, 3], the unsigned int coordinates
        value of each atom.
        - **scalar_f** (Tensor, float32) - [3, ], the 3-D scale factor between
        the real space float coordinates and the unsigned int coordinates.
        - **atom_a** (Tensor, int32) - [M, ], the 1st atom index of each dihedral.
        - **atom_b** (Tensor, int32) - [M, ], the 2nd atom index of each dihedral.
        - **atom_c** (Tensor, int32) - [M, ], the 3rd atom index of each dihedral.
        - **atom_d** (Tensor, int32) - [M, ], the 4th atom index of each dihedral.
        4 atoms are connected in the form a-b-c-d.
        - **ipn** (Tensor, int32) - [M, ], the period of dihedral angle of each dihedral.
        - **pk** (Tensor, float32) - [M, ], the force constant of each dihedral.
        - **gamc** (Tensor, float32) - [M, ], k*cos(phi_0) of each dihedral.
        - **gams** (Tensor, float32) - [M, ], k*sin(phi_0) of each dihedral.
        - **pn** (Tensor, float32) - [M, ], the floating point form of ipn.

    Outputs:
        - **frc_f** (Tensor, float32) - [N, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``

    Examples:
    """

    @prim_attr_register
    def __init__(self, dihedral_numbers):
        self.dihedral_numbers = dihedral_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'atom_d', 'ipn', 'pk',
                                        'gamc', 'gams', 'pn'],
                                outputs=['frc_f'])
        self.add_prim_attr('dihedral_numbers', self.dihedral_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, atom_d_shape,
                    ipn_shape, pk_shape, gamc_shape, gams_shape, pn_shape):
        cls_name = self.name
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_a_shape[0], M, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            atom_c_shape[0], M, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(
            atom_d_shape[0], M, Rel.EQ, "atom_d_shape", cls_name)
        validator.check_int(ipn_shape[0], M, Rel.EQ, "ipn_shape", cls_name)
        validator.check_int(pk_shape[0], M, Rel.EQ, "pk_shape", cls_name)
        validator.check_int(gamc_shape[0], M, Rel.EQ, "gamc_shape", cls_name)
        validator.check_int(gams_shape[0], M, Rel.EQ, "gams_shape", cls_name)
        validator.check_int(pn_shape[0], M, Rel.EQ, "pn_shape", cls_name)
        return uint_crd_f_shape

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, atom_d_type,
                    ipn_type, pk_type, gamc_type, gams_type, pn_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c_type', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_d_type', atom_d_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('ipn_type', ipn_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('pk_type', pk_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gamc_type', gamc_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gams_type', gams_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pn_type', pn_type, [mstype.float32], self.name)
        return pn_type


class DihedralEnergy(PrimitiveWithInfer):
    """
    DihedralEnergy:

    Calculate the potential energy caused by dihedral terms for each 4-atom pair.
    Assume our system has N atoms and M dihedral terms.

    .. math::

        E = k(1 + cos(n*phi - phi_0))

    Inputs:
        Same as operator DihedralForce().

    Outputs:
        - **ene** (Tensor, float32) - [M, ], the potential energy for each
        dihedral term.

    Supported Platforms:
        ``GPU``

    Examples:
    """

    @prim_attr_register
    def __init__(self, dihedral_numbers):
        self.dihedral_numbers = dihedral_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'atom_d', 'ipn', 'pk',
                                        'gamc', 'gams', 'pn'],
                                outputs=['ene'])
        self.add_prim_attr('dihedral_numbers', self.dihedral_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, atom_d_shape,
                    ipn_shape, pk_shape, gamc_shape, gams_shape, pn_shape):
        cls_name = self.name
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_a_shape[0], M, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            atom_c_shape[0], M, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(
            atom_d_shape[0], M, Rel.EQ, "atom_d_shape", cls_name)
        validator.check_int(ipn_shape[0], M, Rel.EQ, "ipn_shape", cls_name)
        validator.check_int(pk_shape[0], M, Rel.EQ, "pk_shape", cls_name)
        validator.check_int(gamc_shape[0], M, Rel.EQ, "gamc_shape", cls_name)
        validator.check_int(gams_shape[0], M, Rel.EQ, "gams_shape", cls_name)
        validator.check_int(pn_shape[0], M, Rel.EQ, "pn_shape", cls_name)
        return [M,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, atom_d_type,
                    ipn_type, pk_type, gamc_type, gams_type, pn_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c_type', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_d_type', atom_d_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('ipn_type', ipn_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('pk_type', pk_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gamc_type', gamc_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gams_type', gams_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pn_type', pn_type, [mstype.float32], self.name)
        return pn_type


class DihedralAtomEnergy(PrimitiveWithInfer):
    """
    DihedralAtomEnergy:

    Add the potential energy caused by dihedral terms to the total potential
    energy of each atom.

    The calculation formula is the same as operator DihedralEnergy().

    Inputs:
        Same as operator DihedralEnergy().

    Outputs:
        - **ene** (Tensor, float32) - [N, ], the accumulated potential
        energy for each atom.

    Supported Platforms:
        ``GPU``

    Examples:
    """

    @prim_attr_register
    def __init__(self, dihedral_numbers):
        self.dihedral_numbers = dihedral_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'atom_d', 'ipn', 'pk',
                                        'gamc', 'gams', 'pn'],
                                outputs=['ene'])
        self.add_prim_attr('dihedral_numbers', self.dihedral_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, atom_d_shape,
                    ipn_shape, pk_shape, gamc_shape, gams_shape, pn_shape):
        cls_name = self.name
        N = uint_crd_f_shape[0]
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_a_shape[0], M, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            atom_c_shape[0], M, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(
            atom_d_shape[0], M, Rel.EQ, "atom_d_shape", cls_name)
        validator.check_int(ipn_shape[0], M, Rel.EQ, "ipn_shape", cls_name)
        validator.check_int(pk_shape[0], M, Rel.EQ, "pk_shape", cls_name)
        validator.check_int(gamc_shape[0], M, Rel.EQ, "gamc_shape", cls_name)
        validator.check_int(gams_shape[0], M, Rel.EQ, "gams_shape", cls_name)
        validator.check_int(pn_shape[0], M, Rel.EQ, "pn_shape", cls_name)
        return [N,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, atom_d_type,
                    ipn_type, pk_type, gamc_type, gams_type, pn_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c_type', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_d_type', atom_d_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('ipn_type', ipn_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('pk_type', pk_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gamc_type', gamc_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gams_type', gams_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pn_type', pn_type, [mstype.float32], self.name)
        return pn_type


class DihedralForceWithAtomEnergy(PrimitiveWithInfer):
    """
    DihedralForceWithAtomEnergy:

    Calculate dihedral force and potential energy together.

    The calculation formula is the same as operator DihedralForce() and DihedralEnergy().

    Inputs:
        Same as operator DihedralForce().

    Outputs:
        - **frc_f** (Tensor, float32) - [N, 3], same as operator DihedralForce().
        - **ene** (Tensor, float32) - [N, ], same as operator DihedralAtomEnergy().

    Supported Platforms:
        ``GPU``

    Examples:
    """

    @prim_attr_register
    def __init__(self, dihedral_numbers):
        self.dihedral_numbers = dihedral_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'atom_d', 'ipn', 'pk',
                                        'gamc', 'gams', 'pn'],
                                outputs=['frc_f', 'ene'])
        self.add_prim_attr('dihedral_numbers', self.dihedral_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, atom_d_shape,
                    ipn_shape, pk_shape, gamc_shape, gams_shape, pn_shape):
        cls_name = self.name
        N = uint_crd_f_shape[0]
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_a_shape[0], M, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            atom_c_shape[0], M, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(
            atom_d_shape[0], M, Rel.EQ, "atom_d_shape", cls_name)
        validator.check_int(ipn_shape[0], M, Rel.EQ, "ipn_shape", cls_name)
        validator.check_int(pk_shape[0], M, Rel.EQ, "pk_shape", cls_name)
        validator.check_int(gamc_shape[0], M, Rel.EQ, "gamc_shape", cls_name)
        validator.check_int(gams_shape[0], M, Rel.EQ, "gams_shape", cls_name)
        validator.check_int(pn_shape[0], M, Rel.EQ, "pn_shape", cls_name)
        return uint_crd_f_shape, [N,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, atom_d_type,
                    ipn_type, pk_type, gamc_type, gams_type, pn_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c_type', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_d_type', atom_d_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('ipn_type', ipn_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('pk_type', pk_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gamc_type', gamc_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gams_type', gams_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pn_type', pn_type, [mstype.float32], self.name)
        return pn_type, pn_type


class AngleForce(PrimitiveWithInfer):
    """
    AngleForce:

    Calculate the force exerted by angles made of 3 atoms on the
    corresponding atoms. Assume the number of angles is M and the
    number of atoms is N.

    .. math::

        dr_{ab} = (x_b-x_a, y_b-y_a, z_b-z_a)
        dr_{cb} = (x_b-x_c, y_b-y_c, z_b-z_c)
        theta = arccos(inner_product(dr_{ab}, dr_{cb})/|dr_{ab}|/|dr_{cb}|)
        F_a = -2*k*(theta-theta_0)/sin(theta)*[cos(theta)/|dr_{ab}|^2*dr_{ab}
            - 1/|dr_{ab}|/|dr_{cb}|*dr_{cb}]
        F_c = -2*k*(theta-theta_0)/sin(theta)*[cos(theta)/|dr_{cb}|^2*dr_{cb}
             - 1/|dr_{cb}|/|dr_{ab}|*dr_{ab}]
        F_b = -F_a - F_c

    Inputs:
        - **uint_crd_f** (Tensor, uint32) - [N, 3], the unsigned int coordinate
        value of each atom.
        - **scaler_f** (Tensor, float32) - [3, ], the 3-D scale factor between
        the real space float coordinates and the unsigned int coordinates.
        - **atom_a** (Tensor, int32) - [M, ], the 1st atom index of each angle.
        - **atom_b** (Tensor, int32) - [M, ], the 2nd and the central atom index
        of each angle.
        - **atom_c** (Tensor, int32) - [M, ], the 3rd atom index of each angle.
        - **angle_k** (Tensor, float32) - [M, ], the force constant for each angle.
        - **angle_theta0** (Tensor, float32) - [M, ], the equilibrium position value
        for each angle.

    Outputs:
        - **frc_f** (Tensor, float32) - [N, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``

    Examples:
    """

    @prim_attr_register
    def __init__(self, angle_numbers):
        self.angle_numbers = angle_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'angle_k',
                                        'angle_theta0'],
                                outputs=['frc_f'])
        self.add_prim_attr('angle_numbers', self.angle_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, angle_k_shape,
                    angle_theta0_shape):
        cls_name = self.name
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_a_shape[0], M, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            atom_c_shape[0], M, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(
            angle_k_shape[0], M, Rel.EQ, "angle_k_shape", cls_name)
        validator.check_int(
            angle_theta0_shape[0], M, Rel.EQ, "angle_theta0_shape", cls_name)
        return uint_crd_f_shape

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, angle_k_type,
                    angle_theta0_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c_type', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('angle_k_type', angle_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('angle_theta0_type', angle_theta0_type, [mstype.float32], self.name)
        return angle_k_type


class AngleEnergy(PrimitiveWithInfer):
    """
    AngleEnergy:

    Calculate the energy caused by 3-atoms angle term.

    .. math::

        dr_{ab} = (x_b-x_a, y_b-y_a, z_b-z_a)
        dr_{cb} = (x_b-x_c, y_b-y_c, z_b-z_c)
        theta = arccos(inner_product(dr_{ab}, dr_{cb})/|dr_{ab}|/|dr_{cb}|)
        E = k*(theta - theta_0)^2

    Inputs:
        Same as operator AngleForce().

    Outputs:
        - **ene** (Tensor, float32) - [M, ], the potential energy for
        each angle term.

    Supported Platforms:
        ``GPU``

    Examples:
    """

    @prim_attr_register
    def __init__(self, angle_numbers):
        self.angle_numbers = angle_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'angle_k',
                                        'angle_theta0'],
                                outputs=['ene'])
        self.add_prim_attr('angle_numbers', self.angle_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, angle_k_shape,
                    angle_theta0_shape):
        cls_name = self.name
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_a_shape[0], M, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            atom_c_shape[0], M, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(
            angle_k_shape[0], M, Rel.EQ, "angle_k_shape", cls_name)
        validator.check_int(
            angle_theta0_shape[0], M, Rel.EQ, "angle_theta0_shape", cls_name)
        return [M,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, angle_k_type,
                    angle_theta0_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c_type', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('angle_k_type', angle_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('angle_theta0_type', angle_theta0_type, [mstype.float32], self.name)
        return angle_k_type


class AngleAtomEnergy(PrimitiveWithInfer):
    """
    AngleAtomEnergy:

    Add the potential energy caused by angle terms to the total potential
    energy of each atom.

    The calculation formula is the same as operator AngleEnergy().

    Inputs:
        Same as operator AngleForce().

    Outputs:
        - **ene** (Tensor, float32) - [N, ], the accumulated potential energy
        for each atom.

    Supported Platforms:
        ``GPU``

    Examples:
    """

    @prim_attr_register
    def __init__(self, angle_numbers):
        self.angle_numbers = angle_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'angle_k',
                                        'angle_theta0'],
                                outputs=['ene'])
        self.add_prim_attr('angle_numbers', self.angle_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, angle_k_shape,
                    angle_theta0_shape):
        cls_name = self.name
        N = uint_crd_f_shape[0]
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_a_shape[0], M, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            atom_c_shape[0], M, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(
            angle_k_shape[0], M, Rel.EQ, "angle_k_shape", cls_name)
        validator.check_int(
            angle_theta0_shape[0], M, Rel.EQ, "angle_theta0_shape", cls_name)
        return [N,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, angle_k_type,
                    angle_theta0_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c_type', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('angle_k_type', angle_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('angle_theta0_type', angle_theta0_type, [mstype.float32], self.name)
        return angle_k_type


class AngleForceWithAtomEnergy(PrimitiveWithInfer):
    """
    AngleForceWithAtomEnergy:

    Calculate angle force and potential energy together.

    The calculation formula is the same as operator AngleForce() and AngleEnergy().

    Inputs:
        Same as operator AngleForce().

    Outputs:
        - **frc_f** (Tensor, float32) - [N, 3], same as operator AngleForce().
        - **ene** (Tensor, float) - [N, ], same as operator AngleAtomEnergy().

    Supported Platforms:
        ``GPU``

    Examples:
    """

    @prim_attr_register
    def __init__(self, angle_numbers):
        self.angle_numbers = angle_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'angle_k',
                                        'angle_theta0'],
                                outputs=['frc_f', 'ene'])
        self.add_prim_attr('angle_numbers', self.angle_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, angle_k_shape,
                    angle_theta0_shape):
        cls_name = self.name
        N = uint_crd_f_shape[0]
        M = atom_a_shape[0]
        validator.check_int(
            uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(
            scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(
            atom_a_shape[0], M, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(
            atom_b_shape[0], M, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(
            atom_c_shape[0], M, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(
            angle_k_shape[0], M, Rel.EQ, "angle_k_shape", cls_name)
        validator.check_int(
            angle_theta0_shape[0], M, Rel.EQ, "angle_theta0_shape", cls_name)
        return uint_crd_f_shape, [N,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, angle_k_type,
                    angle_theta0_type):
        validator.check_tensor_dtype_valid('uint_crd_f_dtype', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f_type', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a_type', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b_type', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c_type', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('angle_k_type', angle_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('angle_theta0_type', angle_theta0_type, [mstype.float32], self.name)
        return angle_k_type, angle_k_type

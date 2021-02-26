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


class BondForce(PrimitiveWithInfer):
    """
    BondForce:

    Calculate the force exerted by the simple harmonic bond on the
    corresponding atoms. Assume the number of harmonic bonds is M and
    the number of atoms is N.

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

    Inputs:
        Same as operator BondForce().

    Outputs:
        - **bond_ene** (Tensor, float32) - [M, 1], The harmonic potential energy
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

    Inputs:
        Same as operator BondForce().

    Outputs:
        - **atom_ene** (Tensor, float32) - [N, 1], he accumulated potential
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

    Inputs:
        Same as operator BondForce().

    Outputs:
        - **frc_f** (Tensor, float32) - [N, 3], Same as operator BondForce().
        - **atom_e** (Tensor, float32) - [N, 1], Same as atom_ene in operator BondAtomEnergy().

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

    Inputs:
        Same as operator BondForce()

    Outputs:
        - **frc_f** (Tensor, float32) - [N, 3], Same as operator BondForce().
        - **atom_v** (Tensor, float32) - [N, 1],The accumulated virial coefficient
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
    """

    @prim_attr_register
    def __init__(self, dihedral_numbers):
        self.dihedral_numbers = dihedral_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'atom_d', 'ipn', 'pk',
                                        'gamc', 'gams', 'pn'],
                                outputs=['frc_f'])
        self.add_prim_attr('dihedral_numbers', self.dihedral_numbers)

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
    """

    @prim_attr_register
    def __init__(self, dihedral_numbers):
        self.dihedral_numbers = dihedral_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'atom_d', 'ipn', 'pk',
                                        'gamc', 'gams', 'pn'],
                                outputs=['ene'])
        self.add_prim_attr('dihedral_numbers', self.dihedral_numbers)

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
    """

    @prim_attr_register
    def __init__(self, dihedral_numbers):
        self.dihedral_numbers = dihedral_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'atom_d', 'ipn', 'pk',
                                        'gamc', 'gams', 'pn'],
                                outputs=['ene'])
        self.add_prim_attr('dihedral_numbers', self.dihedral_numbers)

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
    """

    @prim_attr_register
    def __init__(self, dihedral_numbers):
        self.dihedral_numbers = dihedral_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'atom_d', 'ipn', 'pk',
                                        'gamc', 'gams', 'pn'],
                                outputs=['frc_f', 'ene'])
        self.add_prim_attr('dihedral_numbers', self.dihedral_numbers)

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
    """

    @prim_attr_register
    def __init__(self, angle_numbers):
        self.angle_numbers = angle_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'angle_k',
                                        'angle_theta0'],
                                outputs=['frc_f'])
        self.add_prim_attr('angle_numbers', self.angle_numbers)

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
    """

    @prim_attr_register
    def __init__(self, angle_numbers):
        self.angle_numbers = angle_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'angle_k',
                                        'angle_theta0'],
                                outputs=['ene'])
        self.add_prim_attr('angle_numbers', self.angle_numbers)

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
    """

    @prim_attr_register
    def __init__(self, angle_numbers):
        self.angle_numbers = angle_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'angle_k',
                                        'angle_theta0'],
                                outputs=['ene'])
        self.add_prim_attr('angle_numbers', self.angle_numbers)

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
    """

    @prim_attr_register
    def __init__(self, angle_numbers):
        self.angle_numbers = angle_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'angle_k',
                                        'angle_theta0'],
                                outputs=['frc_f', 'ene'])
        self.add_prim_attr('angle_numbers', self.angle_numbers)

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

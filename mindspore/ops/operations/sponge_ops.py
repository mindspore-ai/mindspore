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

"""
Note:
  SPONGE operators. This is an experimental interface that is subject to change and/or deletion.
"""

import math

from ..primitive import PrimitiveWithInfer, prim_attr_register
from ..._checkparam import Rel
from ..._checkparam import Validator as validator
from ...common import dtype as mstype


class BondForce(PrimitiveWithInfer):
    """
    Calculate the force exerted by the simple harmonic bond on the corresponding atoms.
    Assume the number of harmonic bonds is m and the number of atoms is n.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. math::

        dr = (x_1-x_2, y_1-y_2, z_1-z_2)

    .. math::

        F = (F_x, F_y, F_z) = 2*k*(1 - r_0/|dr|)*dr

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor (x, y, z),
          between the real space float coordinates and the unsigned int coordinates.
          The data type is float32  and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The first atom index of each bond.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The second atom index of each bond.
          The data type is int32 and the shape is :math:`(m,)`.
        - **bond_k** (Tensor) - The force constant of each bond.
          The data type is float32 and the shape is :math:`(m,)`.
        - **bond_r0** (Tensor) - The equlibrium length of each bond.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **frc_f** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, bond_numbers, atom_numbers):
        """Initialize BondForce."""
        validator.check_value_type('bond_numbers', bond_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.bond_numbers = bond_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('bond_numbers', self.bond_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'bond_k', 'bond_r0'],
                                outputs=['frc_f'])

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, bond_k_shape, bond_r0_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.bond_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(bond_k_shape), 1, Rel.EQ, "bond_k_dim", cls_name)
        validator.check_int(len(bond_r0_shape), 1, Rel.EQ, "bond_r0_dim", cls_name)
        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(bond_k_shape[0], m, Rel.EQ, "bond_k_shape", cls_name)
        validator.check_int(bond_r0_shape[0], m, Rel.EQ, "bond_r0_shape", cls_name)
        return uint_crd_f_shape

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, bond_k_type, bond_r0_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('bond_k', bond_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('bond_r0', bond_r0_type, [mstype.float32], self.name)
        return bond_r0_type


class BondEnergy(PrimitiveWithInfer):
    """
    Calculate the harmonic potential energy between each bonded atom pair.
    Assume our system has n atoms and m harmonic bonds.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. math::

        dr = (x_1-x_2, y_1-y_2, z_1-z_2)

    .. math::

        E = k*(|dr| - r_0)^2

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor (x, y, z),
          between the real space float coordinates and the unsigned int coordinates.
          The data type is float32 and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The first atom index of each bond.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The second atom index of each bond.
          The data type is int32 and the shape is :math:`(m,)`.
        - **bond_k** (Tensor) - The force constant of each bond.
          The data type is float32 and the shape is :math:`(m,)`.
        - **bond_r0** (Tensor) - The equlibrium length of each bond.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **bond_ene** (Tensor) - The harmonic potential energy for each bond.
          The data type is float32 and the shape is :math:`(m,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, bond_numbers, atom_numbers):
        """Initialize BondEnergy."""
        validator.check_value_type('bond_numbers', bond_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.bond_numbers = bond_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('bond_numbers', self.bond_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'bond_k', 'bond_r0'],
                                outputs=['bond_ene'])

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, bond_k_shape, bond_r0_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.bond_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(bond_k_shape), 1, Rel.EQ, "bond_k_dim", cls_name)
        validator.check_int(len(bond_r0_shape), 1, Rel.EQ, "bond_r0_dim", cls_name)
        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(bond_k_shape[0], m, Rel.EQ, "bond_k_shape", cls_name)
        validator.check_int(bond_r0_shape[0], m, Rel.EQ, "bond_r0_shape", cls_name)

        return bond_k_shape

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, bond_k_type, bond_r0_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('bond_k', bond_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('bond_r0', bond_r0_type, [mstype.float32], self.name)
        return bond_r0_type


class BondAtomEnergy(PrimitiveWithInfer):
    """
    Add the potential energy caused by simple harmonic bonds to the total
    potential energy of each atom.

    The calculation formula is the same as operator BondEnergy().

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor (x, y, z),
          between the real space float coordinates and the unsigned int coordinates.
          The data type is float32 and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The first atom index of each bond.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The second atom index of each bond.
          The data type is int32 and the shape is :math:`(m,)`.
        - **bond_k** (Tensor) - The force constant of each bond.
          The data type is float32 and the shape is :math:`(m,)`.
        - **bond_r0** (Tensor) - The equlibrium length of each bond.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **atom_ene** (Tensor) - The accumulated potential energy for each atom.
          The data type is float32 and the shape is :math:`(n,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, bond_numbers, atom_numbers):
        """Initialize BondAtomEnergy."""
        validator.check_value_type('bond_numbers', bond_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.bond_numbers = bond_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('bond_numbers', self.bond_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'bond_k', 'bond_r0'],
                                outputs=['atom_ene'])

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, bond_k_shape, bond_r0_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.bond_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(bond_k_shape), 1, Rel.EQ, "bond_k_dim", cls_name)
        validator.check_int(len(bond_r0_shape), 1, Rel.EQ, "bond_r0_dim", cls_name)
        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(bond_k_shape[0], m, Rel.EQ, "bond_k_shape", cls_name)
        validator.check_int(bond_r0_shape[0], m, Rel.EQ, "bond_r0_shape", cls_name)
        return [n,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, bond_k_type, bond_r0_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('bond_k', bond_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('bond_r0', bond_r0_type, [mstype.float32], self.name)
        return bond_r0_type


class BondForceWithAtomEnergy(PrimitiveWithInfer):
    """
    Calculate bond force and harmonic potential energy together.

    The calculation formula is the same as operator BondForce() and BondEnergy().

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor (x, y, z),
          between the real space float coordinates and the unsigned int coordinates.
          The data type is float32 and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The first atom index of each bond.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The second atom index of each bond.
          The data type is int32 and the shape is :math:`(m,)`.
        - **bond_k** (Tensor) - The force constant of each bond.
          The data type is float32 and the shape is :math:`(m,)`.
        - **bond_r0** (Tensor) - The equlibrium length of each bond.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **frc_f** (Tensor, float32) - [n, 3], same as operator BondForce().
        - **atom_e** (Tensor, float32) - [n,], same as atom_ene in operator BondAtomEnergy().

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, bond_numbers, atom_numbers):
        """Initialize BondForceWithAtomEnergy."""
        validator.check_value_type('bond_numbers', bond_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.bond_numbers = bond_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('bond_numbers', self.bond_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'bond_k', 'bond_r0'],
                                outputs=['frc_f', 'atom_e'])

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, bond_k_shape, bond_r0_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.bond_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(bond_k_shape), 1, Rel.EQ, "bond_k_dim", cls_name)
        validator.check_int(len(bond_r0_shape), 1, Rel.EQ, "bond_r0_dim", cls_name)
        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(bond_k_shape[0], m, Rel.EQ, "bond_k_shape", cls_name)
        validator.check_int(bond_r0_shape[0], m, Rel.EQ, "bond_r0_shape", cls_name)

        return uint_crd_f_shape, [n,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, bond_k_type, bond_r0_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)

        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)

        validator.check_tensor_dtype_valid('bond_k', bond_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('bond_r0', bond_r0_type, [mstype.float32], self.name)
        return bond_r0_type, bond_r0_type


class BondForceWithAtomVirial(PrimitiveWithInfer):
    """
    Calculate bond force and the virial coefficient caused by simple harmonic
    bond for each atom together.

    The calculation formula of the force part is the same as operator BondForce().

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    The Virial part is as follows:

    .. math::

        dr = (x_1-x_2, y_1-y_2, z_1-z_2)

    .. math::

        virial = |dr|*(|dr| - r_0)*k

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor (x, y, z),
          between the real space float coordinates and the unsigned int coordinates.
          The data type is float32 and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The first atom index of each bond.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The second atom index of each bond.
          The data type is int32 and the shape is :math:`(m,)`.
        - **bond_k** (Tensor) - The force constant of each bond.
          The data type is float32 and the shape is :math:`(m,)`.
        - **bond_r0** (Tensor) - The equlibrium length of each bond.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **frc_f** (Tensor) - Same as operator BondForce().
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **atom_v** (Tensor) - The accumulated virial coefficient for each atom.
          The data type is float32 and the shape is :math:`(n,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, bond_numbers, atom_numbers):
        """Initialize BondForceWithAtomVirial."""
        validator.check_value_type('bond_numbers', bond_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.bond_numbers = bond_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('bond_numbers', self.bond_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'bond_k', 'bond_r0'],
                                outputs=['frc_f', 'atom_v'])

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, bond_k_shape, bond_r0_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.bond_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(bond_k_shape), 1, Rel.EQ, "bond_k_dim", cls_name)
        validator.check_int(len(bond_r0_shape), 1, Rel.EQ, "bond_r0_dim", cls_name)
        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "uint_crd_f_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(bond_k_shape[0], m, Rel.EQ, "bond_k_shape", cls_name)
        validator.check_int(bond_r0_shape[0], m, Rel.EQ, "bond_r0_shape", cls_name)

        return uint_crd_f_shape, [n,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, bond_k_type, bond_r0_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)

        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)

        validator.check_tensor_dtype_valid('bond_k', bond_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('bond_r0', bond_r0_type, [mstype.float32], self.name)
        return bond_r0_type, bond_r0_type


class DihedralForce(PrimitiveWithInfer):
    """
    Calculate the force exerted by the dihedral term which made of 4-atoms
    on the corresponding atoms. Assume the number of dihedral terms is m and
    the number of atoms is n.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        dihedral_numbers(int32): the number of dihedral terms m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinates
          value of each atom. The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor between
          the real space float coordinates and the unsigned int coordinates.
          The data type is float32 and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The 1st atom index of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The 2nd atom index of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_c** (Tensor) - The 3rd atom index of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_d** (Tensor) - The 4th atom index of each dihedral.
          4 atoms are connected in the form a-b-c-d.
          The data type is int32 and the shape is :math:`(m,)`.
        - **ipn** (Tensor) - The period of dihedral angle of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **pk** (Tensor) - The force constant of each dihedral.
          The data type is float32 and the shape is :math:`(m,)`.
        - **gamc** (Tensor) - k*cos(phi_0) of each dihedral.
          The data type is float32 and the shape is :math:`(m,)`.
        - **gams** (Tensor) - k*sin(phi_0) of each dihedral.
          The data type is float32 and the shape is :math:`(m,)`.
        - **pn** (Tensor) - The floating point form of ipn.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **frc_f** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, dihedral_numbers):
        """Initialize DihedralForce."""
        validator.check_value_type('dihedral_numbers', dihedral_numbers, int, self.name)
        self.dihedral_numbers = dihedral_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'atom_d', 'ipn', 'pk',
                                        'gamc', 'gams', 'pn'],
                                outputs=['frc_f'])
        self.add_prim_attr('dihedral_numbers', self.dihedral_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, atom_d_shape,
                    ipn_shape, pk_shape, gamc_shape, gams_shape, pn_shape):
        cls_name = self.name
        m = self.dihedral_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(atom_c_shape), 1, Rel.EQ, "atom_c_dim", cls_name)
        validator.check_int(len(atom_d_shape), 1, Rel.EQ, "atom_d_dim", cls_name)
        validator.check_int(len(ipn_shape), 1, Rel.EQ, "ipn_dim", cls_name)
        validator.check_int(len(pk_shape), 1, Rel.EQ, "pk_dim", cls_name)
        validator.check_int(len(gamc_shape), 1, Rel.EQ, "gamc_dim", cls_name)
        validator.check_int(len(gams_shape), 1, Rel.EQ, "gams_dim", cls_name)
        validator.check_int(len(pn_shape), 1, Rel.EQ, "pn_dim", cls_name)

        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(atom_c_shape[0], m, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(atom_d_shape[0], m, Rel.EQ, "atom_d_shape", cls_name)
        validator.check_int(ipn_shape[0], m, Rel.EQ, "ipn_shape", cls_name)
        validator.check_int(pk_shape[0], m, Rel.EQ, "pk_shape", cls_name)
        validator.check_int(gamc_shape[0], m, Rel.EQ, "gamc_shape", cls_name)
        validator.check_int(gams_shape[0], m, Rel.EQ, "gams_shape", cls_name)
        validator.check_int(pn_shape[0], m, Rel.EQ, "pn_shape", cls_name)
        return uint_crd_f_shape

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, atom_d_type,
                    ipn_type, pk_type, gamc_type, gams_type, pn_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_d', atom_d_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('ipn', ipn_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('pk', pk_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gamc', gamc_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gams', gams_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pn', pn_type, [mstype.float32], self.name)

        return pn_type


class DihedralEnergy(PrimitiveWithInfer):
    """
    Calculate the potential energy caused by dihedral terms for each 4-atom pair.
    Assume our system has n atoms and m dihedral terms.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        dihedral_numbers(int32): the number of dihedral terms m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinates
          value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor between
          the real space float coordinates and the unsigned int coordinates.
          The data type is float32 and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The 1st atom index of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The 2nd atom index of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_c** (Tensor) - The 3rd atom index of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_d** (Tensor) - The 4th atom index of each dihedral.
          4 atoms are connected in the form a-b-c-d.
          The data type is int32 and the shape is :math:`(m,)`.
        - **ipn** (Tensor) - The period of dihedral angle of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **pk** (Tensor) - The force constant of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **gamc** (Tensor) - k*cos(phi_0) of each dihedral.
          The data type is float32 and the shape is :math:`(m,)`.
        - **gams** (Tensor) - k*sin(phi_0) of each dihedral.
          The data type is float32 and the shape is :math:`(m,)`.
        - **pn** (Tensor) - The floating point form of ipn.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **ene** (Tensor) - The potential energy for each
          dihedral term. The data type is float32 and the shape is :math:`(m,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, dihedral_numbers):
        """Initialize DihedralEnergy."""
        validator.check_value_type('dihedral_numbers', dihedral_numbers, int, self.name)
        self.dihedral_numbers = dihedral_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'atom_d', 'ipn', 'pk',
                                        'gamc', 'gams', 'pn'],
                                outputs=['ene'])
        self.add_prim_attr('dihedral_numbers', self.dihedral_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, atom_d_shape,
                    ipn_shape, pk_shape, gamc_shape, gams_shape, pn_shape):
        cls_name = self.name
        m = self.dihedral_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(atom_c_shape), 1, Rel.EQ, "atom_c_dim", cls_name)
        validator.check_int(len(atom_d_shape), 1, Rel.EQ, "atom_d_dim", cls_name)
        validator.check_int(len(ipn_shape), 1, Rel.EQ, "ipn_dim", cls_name)
        validator.check_int(len(pk_shape), 1, Rel.EQ, "pk_dim", cls_name)
        validator.check_int(len(gamc_shape), 1, Rel.EQ, "gamc_dim", cls_name)
        validator.check_int(len(gams_shape), 1, Rel.EQ, "gams_dim", cls_name)
        validator.check_int(len(pn_shape), 1, Rel.EQ, "pn_dim", cls_name)

        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(atom_c_shape[0], m, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(atom_d_shape[0], m, Rel.EQ, "atom_d_shape", cls_name)
        validator.check_int(ipn_shape[0], m, Rel.EQ, "ipn_shape", cls_name)
        validator.check_int(pk_shape[0], m, Rel.EQ, "pk_shape", cls_name)
        validator.check_int(gamc_shape[0], m, Rel.EQ, "gamc_shape", cls_name)
        validator.check_int(gams_shape[0], m, Rel.EQ, "gams_shape", cls_name)
        validator.check_int(pn_shape[0], m, Rel.EQ, "pn_shape", cls_name)
        return [m,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, atom_d_type,
                    ipn_type, pk_type, gamc_type, gams_type, pn_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_d', atom_d_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('ipn', ipn_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('pk', pk_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gamc', gamc_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gams', gams_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pn', pn_type, [mstype.float32], self.name)

        return pn_type


class DihedralAtomEnergy(PrimitiveWithInfer):
    """
    Add the potential energy caused by dihedral terms to the total potential
    energy of each atom.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    The calculation formula is the same as operator DihedralEnergy().

    Args:
        dihedral_numbers(int32): the number of dihedral terms m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinates
          value of each atom. The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor between
          the real space float coordinates and the unsigned int coordinates.
          The data type is float32 and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The 1st atom index of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The 2nd atom index of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_c** (Tenso) - The 3rd atom index of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_d** (Tensor) - The 4th atom index of each dihedral.
          4 atoms are connected in the form a-b-c-d. The data type is int32 and the shape is :math:`(m,)`.
        - **ipn** (Tensor) - The period of dihedral angle of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **pk** (Tensor) - The force constant of each dihedral.
          The data type is float32 and the shape is :math:`(m,)`.
        - **gamc** (Tensor) - k*cos(phi_0) of each dihedral.
          The data type is float32 and the shape is :math:`(m,)`.
        - **gams** (Tensor) - k*sin(phi_0) of each dihedral.
          The data type is float32 and the shape is :math:`(m,)`.
        - **pn** (Tensor) - The floating point form of ipn.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **ene** (Tensor) - The accumulated potential
          energy for each atom. The data type is float32 and the shape is :math:`(n,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, dihedral_numbers):
        """Initialize DihedralAtomEnergy."""
        validator.check_value_type('dihedral_numbers', dihedral_numbers, int, self.name)
        self.dihedral_numbers = dihedral_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'atom_d', 'ipn', 'pk',
                                        'gamc', 'gams', 'pn'],
                                outputs=['ene'])
        self.add_prim_attr('dihedral_numbers', self.dihedral_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, atom_d_shape,
                    ipn_shape, pk_shape, gamc_shape, gams_shape, pn_shape):
        cls_name = self.name
        n = uint_crd_f_shape[0]
        m = self.dihedral_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(atom_c_shape), 1, Rel.EQ, "atom_c_dim", cls_name)
        validator.check_int(len(atom_d_shape), 1, Rel.EQ, "atom_d_dim", cls_name)
        validator.check_int(len(ipn_shape), 1, Rel.EQ, "ipn_dim", cls_name)
        validator.check_int(len(pk_shape), 1, Rel.EQ, "pk_dim", cls_name)
        validator.check_int(len(gamc_shape), 1, Rel.EQ, "gamc_dim", cls_name)
        validator.check_int(len(gams_shape), 1, Rel.EQ, "gams_dim", cls_name)
        validator.check_int(len(pn_shape), 1, Rel.EQ, "pn_dim", cls_name)

        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(atom_c_shape[0], m, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(atom_d_shape[0], m, Rel.EQ, "atom_d_shape", cls_name)
        validator.check_int(ipn_shape[0], m, Rel.EQ, "ipn_shape", cls_name)
        validator.check_int(pk_shape[0], m, Rel.EQ, "pk_shape", cls_name)
        validator.check_int(gamc_shape[0], m, Rel.EQ, "gamc_shape", cls_name)
        validator.check_int(gams_shape[0], m, Rel.EQ, "gams_shape", cls_name)
        validator.check_int(pn_shape[0], m, Rel.EQ, "pn_shape", cls_name)
        return [n,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, atom_d_type,
                    ipn_type, pk_type, gamc_type, gams_type, pn_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_d', atom_d_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('ipn', ipn_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('pk', pk_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gamc', gamc_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gams', gams_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pn', pn_type, [mstype.float32], self.name)

        return pn_type


class DihedralForceWithAtomEnergy(PrimitiveWithInfer):
    """
    Calculate dihedral force and potential energy together.

    The calculation formula is the same as operator DihedralForce() and DihedralEnergy().

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        dihedral_numbers(int32): the number of dihedral terms m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinates
          value of each atom. The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor between
          the real space float coordinates and the unsigned int coordinates.
          The data type is float32 and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The 1st atom index of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The 2nd atom index of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_c** (Tenso) - The 3rd atom index of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_d** (Tensor) - The 4th atom index of each dihedral.
          4 atoms are connected in the form a-b-c-d. The data type is int32 and the shape is :math:`(m,)`.
        - **ipn** (Tensor) - The period of dihedral angle of each dihedral.
          The data type is int32 and the shape is :math:`(m,)`.
        - **pk** (Tensor) - The force constant of each dihedral.
          The data type is float32 and the shape is :math:`(m,)`.
        - **gamc** (Tensor) - k*cos(phi_0) of each dihedral.
          The data type is float32 and the shape is :math:`(m,)`.
        - **gams** (Tensor) - k*sin(phi_0) of each dihedral.
          The data type is float32 and the shape is :math:`(m,)`.
        - **pn** (Tensor) - The floating point form of ipn.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **frc_f** (Tensor) - Same as operator DihedralForce().
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **ene** (Tensor) - Same as operator DihedralAtomEnergy().
          The data type is float32 and the shape is :math:`(n,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, dihedral_numbers):
        """Initialize DihedralForceWithAtomEnergy."""
        validator.check_value_type('dihedral_numbers', dihedral_numbers, int, self.name)
        self.dihedral_numbers = dihedral_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'atom_d', 'ipn', 'pk',
                                        'gamc', 'gams', 'pn'],
                                outputs=['frc_f', 'ene'])
        self.add_prim_attr('dihedral_numbers', self.dihedral_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, atom_d_shape,
                    ipn_shape, pk_shape, gamc_shape, gams_shape, pn_shape):
        cls_name = self.name
        n = uint_crd_f_shape[0]
        m = self.dihedral_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(atom_c_shape), 1, Rel.EQ, "atom_c_dim", cls_name)
        validator.check_int(len(atom_d_shape), 1, Rel.EQ, "atom_d_dim", cls_name)
        validator.check_int(len(ipn_shape), 1, Rel.EQ, "ipn_dim", cls_name)
        validator.check_int(len(pk_shape), 1, Rel.EQ, "pk_dim", cls_name)
        validator.check_int(len(gamc_shape), 1, Rel.EQ, "gamc_dim", cls_name)
        validator.check_int(len(gams_shape), 1, Rel.EQ, "gams_dim", cls_name)
        validator.check_int(len(pn_shape), 1, Rel.EQ, "pn_dim", cls_name)

        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(atom_c_shape[0], m, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(atom_d_shape[0], m, Rel.EQ, "atom_d_shape", cls_name)
        validator.check_int(ipn_shape[0], m, Rel.EQ, "ipn_shape", cls_name)
        validator.check_int(pk_shape[0], m, Rel.EQ, "pk_shape", cls_name)
        validator.check_int(gamc_shape[0], m, Rel.EQ, "gamc_shape", cls_name)
        validator.check_int(gams_shape[0], m, Rel.EQ, "gams_shape", cls_name)
        validator.check_int(pn_shape[0], m, Rel.EQ, "pn_shape", cls_name)
        return uint_crd_f_shape, [n,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, atom_d_type,
                    ipn_type, pk_type, gamc_type, gams_type, pn_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_d', atom_d_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('ipn', ipn_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('pk', pk_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gamc', gamc_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('gams', gams_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pn', pn_type, [mstype.float32], self.name)

        return pn_type, pn_type


class AngleForce(PrimitiveWithInfer):
    """
    Calculate the force exerted by angles made of 3 atoms on the
    corresponding atoms. Assume the number of angles is m and the
    number of atoms is n.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. math::
        dr_{ab} = (x_b-x_a, y_b-y_a, z_b-z_a)
    .. math::
        dr_{cb} = (x_b-x_c, y_b-y_c, z_b-z_c)
    .. math::
        theta = arccos(inner_product(dr_{ab}, dr_{cb})/|dr_{ab}|/|dr_{cb}|)
    .. math::
        F_a = -2*k*(theta-theta_0)/sin(theta)*[cos(theta)/|dr_{ab}|^2*dr_{ab}
            - 1/|dr_{ab}|/|dr_{cb}|*dr_{cb}]
    .. math::
        F_c = -2*k*(theta-theta_0)/sin(theta)*[cos(theta)/|dr_{cb}|^2*dr_{cb}
            - 1/|dr_{cb}|/|dr_{ab}|*dr_{ab}]
    .. math::
        F_b = -F_a - F_c

    Args:
        angle_numbers(int32): the number of angles m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor between
          the real space float coordinates and the unsigned int coordinates.
          The data type is float32 and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The 1st atom index of each angle.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The 2nd and the central atom index of each angle.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_c** (Tensor) - The 3rd atom index of each angle.
          The data type is int32 and the shape is :math:`(m,)`.
        - **angle_k** (Tensor) - The force constant for each angle.
          The data type is float32 and the shape is :math:`(m,)`.
        - **angle_theta0** (Tensor) - The equilibrium position value for each angle.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **frc_f** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, angle_numbers):
        """Initialize AngleForce."""
        validator.check_value_type('angle_numbers', angle_numbers, int, self.name)
        self.angle_numbers = angle_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'angle_k',
                                        'angle_theta0'],
                                outputs=['frc_f'])
        self.add_prim_attr('angle_numbers', self.angle_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, angle_k_shape,
                    angle_theta0_shape):
        cls_name = self.name
        m = self.angle_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(atom_c_shape), 1, Rel.EQ, "atom_c_dim", cls_name)
        validator.check_int(len(angle_k_shape), 1, Rel.EQ, "angle_k_dim", cls_name)
        validator.check_int(len(angle_theta0_shape), 1, Rel.EQ, "angle_theta0_dim", cls_name)

        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(atom_c_shape[0], m, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(angle_k_shape[0], m, Rel.EQ, "angle_k_shape", cls_name)
        validator.check_int(angle_theta0_shape[0], m, Rel.EQ, "angle_theta0_shape", cls_name)
        return uint_crd_f_shape

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, angle_k_type,
                    angle_theta0_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('angle_k', angle_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('angle_theta0', angle_theta0_type, [mstype.float32], self.name)
        return angle_k_type


class AngleEnergy(PrimitiveWithInfer):
    """
    Calculate the energy caused by 3-atoms angle term. Assume the number of angles is m and the
    number of atoms is n.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. math::
        dr_{ab} = (x_b-x_a, y_b-y_a, z_b-z_a)
    .. math::
        dr_{cb} = (x_b-x_c, y_b-y_c, z_b-z_c)
    .. math::
        theta = arccos(inner_product(dr_{ab}, dr_{cb})/|dr_{ab}|/|dr_{cb}|)
    .. math::
        E = k*(theta - theta_0)^2

    Args:
        angle_numbers(int32): the number of angles m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor between
          the real space float coordinates and the unsigned int coordinates.
          The data type is float32 and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The 1st atom index of each angle.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The 2nd and the central atom index of each angle.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_c** (Tensor) - The 3rd atom index of each angle.
          The data type is int32 and the shape is :math:`(m,)`.
        - **angle_k** (Tensor) - The force constant for each angle.
          The data type is float32 and the shape is :math:`(m,)`.
        - **angle_theta0** (Tensor) - The equilibrium position value for each angle.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **ene** (Tensor) - The potential energy for each angle term.
          The data type is float32 and the shape is :math:`(m,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, angle_numbers):
        """Initialize AngleEnergy."""
        validator.check_value_type('angle_numbers', angle_numbers, int, self.name)
        self.angle_numbers = angle_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'angle_k',
                                        'angle_theta0'],
                                outputs=['ene'])
        self.add_prim_attr('angle_numbers', self.angle_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, angle_k_shape,
                    angle_theta0_shape):
        cls_name = self.name
        m = self.angle_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(atom_c_shape), 1, Rel.EQ, "atom_c_dim", cls_name)
        validator.check_int(len(angle_k_shape), 1, Rel.EQ, "angle_k_dim", cls_name)
        validator.check_int(len(angle_theta0_shape), 1, Rel.EQ, "angle_theta0_dim", cls_name)

        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(atom_c_shape[0], m, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(angle_k_shape[0], m, Rel.EQ, "angle_k_shape", cls_name)
        validator.check_int(angle_theta0_shape[0], m, Rel.EQ, "angle_theta0_shape", cls_name)
        return [m,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, angle_k_type,
                    angle_theta0_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('angle_k', angle_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('angle_theta0', angle_theta0_type, [mstype.float32], self.name)
        return angle_k_type


class AngleAtomEnergy(PrimitiveWithInfer):
    """
    Add the potential energy caused by angle terms to the total potential
    energy of each atom. Assume the number of angles is m and the
    number of atoms is n.

    The calculation formula is the same as operator AngleEnergy().

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        angle_numbers(int32): the number of angles m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor between
          the real space float coordinates and the unsigned int coordinates.
          The data type is float32 and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The 1st atom index of each angle.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The 2nd and the central atom index of each angle.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_c** (Tensor) - The 3rd atom index of each angle.
          The data type is int32 and the shape is :math:`(m,)`.
        - **angle_k** (Tensor) - The force constant for each angle.
          The data type is float32 and the shape is :math:`(m,)`.
        - **angle_theta0** (Tensor) - The equilibrium position value for each angle.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **ene** (Tensor) - The accumulated potential energy for each atom.
          The data type is float32 and the shape is :math:`(n,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, angle_numbers):
        """Initialize AngleAtomEnergy."""
        validator.check_value_type('angle_numbers', angle_numbers, int, self.name)
        self.angle_numbers = angle_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'angle_k',
                                        'angle_theta0'],
                                outputs=['ene'])
        self.add_prim_attr('angle_numbers', self.angle_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, angle_k_shape,
                    angle_theta0_shape):
        cls_name = self.name
        n = uint_crd_f_shape[0]
        m = self.angle_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(atom_c_shape), 1, Rel.EQ, "atom_c_dim", cls_name)
        validator.check_int(len(angle_k_shape), 1, Rel.EQ, "angle_k_dim", cls_name)
        validator.check_int(len(angle_theta0_shape), 1, Rel.EQ, "angle_theta0_dim", cls_name)

        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(atom_c_shape[0], m, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(angle_k_shape[0], m, Rel.EQ, "angle_k_shape", cls_name)
        validator.check_int(angle_theta0_shape[0], m, Rel.EQ, "angle_theta0_shape", cls_name)
        return [n,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, angle_k_type,
                    angle_theta0_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('angle_k', angle_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('angle_theta0', angle_theta0_type, [mstype.float32], self.name)
        return angle_k_type


class AngleForceWithAtomEnergy(PrimitiveWithInfer):
    """
    Calculate angle force and potential energy together. Assume the number of angles is m and the
    number of atoms is n.

    The calculation formula is the same as operator AngleForce() and AngleEnergy().

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        angle_numbers(int32): the number of angles m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor between
          the real space float coordinates and the unsigned int coordinates.
          The data type is float and the shape is :math:`(3,)`.
        - **atom_a** (Tensor) - The 1st atom index of each angle.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_b** (Tensor) - The 2nd and the central atom index of each angle.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_c** (Tensor) - The 3rd atom index of each angle.
          The data type is int32 and the shape is :math:`(m,)`.
        - **angle_k** (Tensor) - The force constant for each angle.
          The data type is float32 and the shape is :math:`(m,)`.
        - **angle_theta0** (Tensor) - The equilibrium position value for each angle.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **frc_f** (Tensor) - same as operator AngleForce().
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **ene** (Tensor) - same as operator AngleAtomEnergy().
          The data type is float and the shape is :math:`(n,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, angle_numbers):
        """Initialize AngleForceWithAtomEnergy."""
        validator.check_value_type('angle_numbers', angle_numbers, int, self.name)
        self.angle_numbers = angle_numbers
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'atom_c', 'angle_k',
                                        'angle_theta0'],
                                outputs=['frc_f', 'ene'])
        self.add_prim_attr('angle_numbers', self.angle_numbers)

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, atom_a_shape, atom_b_shape, atom_c_shape, angle_k_shape,
                    angle_theta0_shape):
        cls_name = self.name
        n = uint_crd_f_shape[0]
        m = self.angle_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(len(atom_a_shape), 1, Rel.EQ, "atom_a_dim", cls_name)
        validator.check_int(len(atom_b_shape), 1, Rel.EQ, "atom_b_dim", cls_name)
        validator.check_int(len(atom_c_shape), 1, Rel.EQ, "atom_c_dim", cls_name)
        validator.check_int(len(angle_k_shape), 1, Rel.EQ, "angle_k_dim", cls_name)
        validator.check_int(len(angle_theta0_shape), 1, Rel.EQ, "angle_theta0_dim", cls_name)

        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "atom_a_shape", cls_name)
        validator.check_int(atom_b_shape[0], m, Rel.EQ, "atom_b_shape", cls_name)
        validator.check_int(atom_c_shape[0], m, Rel.EQ, "atom_c_shape", cls_name)
        validator.check_int(angle_k_shape[0], m, Rel.EQ, "angle_k_shape", cls_name)
        validator.check_int(angle_theta0_shape[0], m, Rel.EQ, "angle_theta0_shape", cls_name)
        return uint_crd_f_shape, [n,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, atom_c_type, angle_k_type,
                    angle_theta0_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_c', atom_c_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('angle_k', angle_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('angle_theta0', angle_theta0_type, [mstype.float32], self.name)
        return angle_k_type, angle_k_type


class Dihedral14LJForce(PrimitiveWithInfer):
    """
    Calculate the Lennard-Jones part of 1,4 dihedral force correction
    for each necessary dihedral terms on the corresponding atoms.

    Assume the number of necessary dihedral 1,4 terms is m, the number of atoms is n,
    and the number of Lennard-Jones types for all atoms is P, which means
    there will be q = P*(P+1)/2 types of possible Lennard-Jones interactions
    for all kinds of atom pairs.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. math::
        dr = (x_a-x_b, y_a-y_b, z_a-z_b)
    .. math::
        F = k*(-12*A/|dr|^{14} + 6*B/|dr|^{8})*dr

    Args:
        nb14_numbers (int32): the number of necessary dihedral 1,4 terms m.
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **LJ_type** (Tensor) - The Lennard-Jones type of each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **charge** (Tensor) - The charge of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **boxlength_f** (Tensor) - The length of molecular simulation box in 3 dimensions.
          The data type is float32 and the shape is :math:`(3,)`.
        - **a_14** (Tensor) - The first atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **b_14** (Tensor) - The second atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **lj_scale_factor** (Tensor) - The scale factor for the
          Lennard-Jones part of force correction of each dihedral 1,4 term.
          The data type is float32 and the shape is :math:`(m,)`.
        - **LJ_type_A** (Tensor) - The A parameter in Lennard-Jones scheme of each atom pair type.
          q is the number of atom pair. The data type is float32 and the shape is :math:`(q,)`.
        - **LJ_type_B** (Tensor) - The B parameter in Lennard-Jones shceme of each atom pair type.
          q is the number of atom pair. The data type is float32 and the shape is :math:`(q,)`.

    Outputs:
        - **frc_f** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, nb14_numbers, atom_numbers):
        """Initialize Dihedral14LJForce."""
        validator.check_value_type('nb14_numbers', nb14_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.dihedral_14_numbers = nb14_numbers
        self.atom_numbers = atom_numbers
        self.init_prim_io_names(
            inputs=['uint_crd_f', 'LJtype', 'charge', 'boxlength_f', 'a_14', 'b_14', 'lj_scale_factor',
                    'LJ_type_A', 'LJ_type_B'],
            outputs=['frc_f'])
        self.add_prim_attr('dihedral_14_numbers', self.dihedral_14_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)

    def infer_shape(self, uint_crd_f_shape, ljtype_shape, charge_shape, boxlength_f_shape, a_14_shape, b_14_shape,
                    lj_scale_factor_shape, lj_type_a_shape, lj_type_b_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.dihedral_14_numbers
        q = lj_type_a_shape[0]
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(ljtype_shape), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge_shape), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(boxlength_f_shape), 1, Rel.EQ, "boxlength_f_dim", cls_name)
        validator.check_int(len(a_14_shape), 1, Rel.EQ, "a_14_dim", cls_name)
        validator.check_int(len(b_14_shape), 1, Rel.EQ, "b_14_dim", cls_name)
        validator.check_int(len(lj_scale_factor_shape), 1, Rel.EQ, "lj_scale_factor_dim", cls_name)
        validator.check_int(len(lj_type_b_shape), 1, Rel.EQ, "LJ_type_B_dim", cls_name)

        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f[1]", cls_name)
        validator.check_int(ljtype_shape[0], n, Rel.EQ, "LJtype", cls_name)
        validator.check_int(charge_shape[0], n, Rel.EQ, "charge", cls_name)
        validator.check_int(boxlength_f_shape[0], 3, Rel.EQ, "boxlength_f", cls_name)
        validator.check_int(lj_type_b_shape[0], q, Rel.EQ, "LJ_type_B", cls_name)
        validator.check_int(a_14_shape[0], m, Rel.EQ, "a_14_shape", cls_name)
        validator.check_int(b_14_shape[0], m, Rel.EQ, "b_14_shape", cls_name)
        validator.check_int(lj_scale_factor_shape[0], m, Rel.EQ, "lj_scale_factor_shape", cls_name)
        return uint_crd_f_shape

    def infer_dtype(self, uint_crd_f_dtype, ljtype_dtype, charge_dtype, boxlength_f_type, a_14_type, b_14_type,
                    lj_scale_factor_type, lj_type_a_type, lj_type_b_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('boxlength_f', boxlength_f_type, [mstype.float32], self.name)

        validator.check_tensor_dtype_valid('a_14', a_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('b_14', b_14_type, [mstype.int32], self.name)

        validator.check_tensor_dtype_valid('lj_scale_factor', lj_scale_factor_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('LJ_type_A', lj_type_a_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('LJ_type_B', lj_type_b_type, [mstype.float32], self.name)
        return lj_type_b_type


class Dihedral14LJEnergy(PrimitiveWithInfer):
    """
    Calculate the Lennard-Jones part of 1,4 dihedral energy correction for
    each necessary dihedral terms on the corresponding atoms.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. math::
        dr = (x_a-x_b, y_a-y_b, z_a-z-b)
    .. math::
        E = k*(A/|dr|^{12} - B/|dr|^{6})

    Args:
        nb14_numbers (int32): the number of necessary dihedral 1,4 terms m.
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **LJ_type** (Tensor) - The Lennard-Jones type of each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **charge** (Tensor) - The charge of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **boxlength_f** (Tensor) - The length of molecular simulation box in 3 dimensions.
          The data type is float32 and the shape is :math:`(3,)`.
        - **a_14** (Tensor) - The first atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **b_14** (Tensor) - The second atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **lj_scale_factor** (Tensor) - The scale factor for the
          Lennard-Jones part of force correction of each dihedral 1,4 term.
          The data type is float32 and the shape is :math:`(m,)`.
        - **LJ_type_A** (Tensor) - The A parameter in Lennard-Jones scheme of each atom pair type.
          q is the number of atom pair. The data type is float32 and the shape is :math:`(q,)`.
        - **LJ_type_B** (Tensor) - The B parameter in Lennard-Jones shceme of each atom pair type.
          q is the number of atom pair. The data type is float32 and the shape is :math:`(q,)`.

    Outputs:
        - **ene** (Tensor) - The Lennard-Jones potential energy correction.
          The data type is float32 and the shape is :math:`(m,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, nb14_numbers, atom_numbers):
        """Initialize Dihedral14LJEnergy"""
        validator.check_value_type('nb14_numbers', nb14_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.dihedral_14_numbers = nb14_numbers
        self.atom_numbers = atom_numbers

        self.init_prim_io_names(
            inputs=['uint_crd_f', 'LJtype', 'charge', 'boxlength_f', 'a_14', 'b_14', 'lj_scale_factor',
                    'LJ_type_A', 'LJ_type_B'],
            outputs=['ene'])
        self.add_prim_attr('dihedral_14_numbers', self.dihedral_14_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)

    def infer_shape(self, uint_crd_f_shape, ljtype_shape, charge_shape, boxlength_f_shape, a_14_shape, b_14_shape,
                    lj_scale_factor_shape, lj_type_a_shape, lj_type_b_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.dihedral_14_numbers
        q = lj_type_a_shape[0]
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(ljtype_shape), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge_shape), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(boxlength_f_shape), 1, Rel.EQ, "boxlength_f_dim", cls_name)
        validator.check_int(len(a_14_shape), 1, Rel.EQ, "a_14_dim", cls_name)
        validator.check_int(len(b_14_shape), 1, Rel.EQ, "b_14_dim", cls_name)
        validator.check_int(len(lj_scale_factor_shape), 1, Rel.EQ, "lj_scale_factor_dim", cls_name)
        validator.check_int(len(lj_type_b_shape), 1, Rel.EQ, "LJ_type_B_dim", cls_name)

        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f[1]", cls_name)
        validator.check_int(ljtype_shape[0], n, Rel.EQ, "LJtype", cls_name)
        validator.check_int(charge_shape[0], n, Rel.EQ, "charge", cls_name)
        validator.check_int(boxlength_f_shape[0], 3, Rel.EQ, "boxlength_f", cls_name)
        validator.check_int(lj_type_b_shape[0], q, Rel.EQ, "LJ_type_B", cls_name)
        validator.check_int(a_14_shape[0], m, Rel.EQ, "a_14_shape", cls_name)
        validator.check_int(b_14_shape[0], m, Rel.EQ, "b_14_shape", cls_name)
        validator.check_int(lj_scale_factor_shape[0], m, Rel.EQ, "lj_scale_factor_shape", cls_name)
        return [self.dihedral_14_numbers,]

    def infer_dtype(self, uint_crd_f_dtype, ljtype_dtype, charge_dtype, boxlength_f_type, a_14_type, b_14_type,
                    lj_scale_factor_type, lj_type_a_type, lj_type_b_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('boxlength_f', boxlength_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('a_14', a_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('b_14', b_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('lj_scale_factor', lj_scale_factor_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('LJ_type_A', lj_type_a_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('LJ_type_B', lj_type_b_type, [mstype.float32], self.name)

        return lj_type_a_type


class Dihedral14LJForceWithDirectCF(PrimitiveWithInfer):
    """
    Calculate the Lennard-Jones part and the Coulomb part of force correction
    for each necessary dihedral 1,4 terms.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    The calculation formula of the Lennard-Jones part is the same as operator
    Dihedral14LJForce(), and the Coulomb part is as follows:

    .. math::
            dr = (x_a-x_b, y_a-y_b, z_a-z_b)
    .. math::
            F = -k*q_a*q_b/|r|^3*dr

    Args:
        nb14_numbers (int32): the number of necessary dihedral 1,4 terms m.
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **LJ_type** (Tensor) - The Lennard-Jones type of each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **charge** (Tensor) - The charge of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **boxlength_f** (Tensor) - The length of molecular simulation box in 3 dimensions.
          The data type is float32 and the shape is :math:`(3,)`.
        - **a_14** (Tensor) - The first atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **b_14** (Tensor) - The second atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **lj_scale_factor** (Tensor) - The scale factor for the
          Lennard-Jones part of force correction of each dihedral 1,4 term.
          The data type is float32 and the shape is :math:`(m,)`.
        - **cf_scale_factor** (Tensor) - The scale factor for the
          Coulomb part of force correction for each dihedral 1,4 terms.
          The data type is float and the shape is :math:`(m,)`.
        - **LJ_type_A** (Tensor) - The A parameter in Lennard-Jones scheme of each atom pair type.
          q is the number of atom pair. The data type is float32 and the shape is :math:`(q,)`.
        - **LJ_type_B** (Tensor) - The B parameter in Lennard-Jones shceme of each atom pair type.
          q is the number of atom pair. The data type is float32 and the shape is :math:`(q,)`.

    Outputs:
        - **frc_f** (Tensor, float) - [n, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, nb14_numbers, atom_numbers):
        """Initialize Dihedral14LJForceWithDirectCF."""
        validator.check_value_type('nb14_numbers', nb14_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.dihedral_14_numbers = nb14_numbers
        self.atom_numbers = atom_numbers

        self.init_prim_io_names(
            inputs=['uint_crd_f', 'LJtype', 'charge', 'boxlength_f', 'a_14', 'b_14', 'lj_scale_factor',
                    'cf_scale_factor',
                    'LJ_type_A', 'LJ_type_B'],
            outputs=['frc_f'])
        self.add_prim_attr('dihedral_14_numbers', self.dihedral_14_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)

    def infer_shape(self, uint_crd_f_shape, ljtype_shape, charge_shape, boxlength_f_shape, a_14_shape, b_14_shape,
                    lj_scale_factor_shape, cf_scale_factor_shape, lj_type_a_shape, lj_type_b_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.dihedral_14_numbers
        q = lj_type_a_shape[0]
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(ljtype_shape), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge_shape), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(boxlength_f_shape), 1, Rel.EQ, "boxlength_f_dim", cls_name)
        validator.check_int(len(a_14_shape), 1, Rel.EQ, "a_14_dim", cls_name)
        validator.check_int(len(b_14_shape), 1, Rel.EQ, "b_14_dim", cls_name)
        validator.check_int(len(lj_scale_factor_shape), 1, Rel.EQ, "lj_scale_factor_dim", cls_name)
        validator.check_int(len(cf_scale_factor_shape), 1, Rel.EQ, "cf_scale_factor_dim", cls_name)
        validator.check_int(len(lj_type_b_shape), 1, Rel.EQ, "LJ_type_B_dim", cls_name)

        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(ljtype_shape[0], n, Rel.EQ, "LJtype_shape", cls_name)
        validator.check_int(charge_shape[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(boxlength_f_shape[0], 3, Rel.EQ, "boxlength_f_shape", cls_name)
        validator.check_int(lj_type_b_shape[0], q, Rel.EQ, "LJ_type_B_shape", cls_name)
        validator.check_int(a_14_shape[0], m, Rel.EQ, "a_14_shape", cls_name)
        validator.check_int(b_14_shape[0], m, Rel.EQ, "b_14_shape", cls_name)
        validator.check_int(lj_scale_factor_shape[0], m, Rel.EQ, "lj_scale_factor_shape", cls_name)
        validator.check_int(cf_scale_factor_shape[0], m, Rel.EQ, "cf_scale_factor_shape", cls_name)
        return [self.atom_numbers, 3]

    def infer_dtype(self, uint_crd_f_dtype, ljtype_dtype, charge_dtype, boxlength_f_type, a_14_type, b_14_type,
                    lj_scale_factor_type, cf_scale_factor_type, lj_type_a_type, lj_type_b_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('boxlength_f', boxlength_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('a_14', a_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('b_14', b_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('lj_scale_factor', lj_scale_factor_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('cf_scale_factor', cf_scale_factor_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('LJ_type_A', lj_type_a_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('LJ_type_B', lj_type_b_type, [mstype.float32], self.name)

        return lj_type_a_type


class Dihedral14LJCFForceWithAtomEnergy(PrimitiveWithInfer):
    """
    Calculate the Lennard-Jones and Coulumb energy correction and force correction
    for each necessary dihedral 1,4 terms together and add them to the total force
    and potential energy for each atom.

    The calculation formula of force correction is the same as operator
    :class:`Dihedral14LJForceWithDirectCF`, and the energy correction part is the same
    as operator :class:`Dihedral14LJEnergy` and :class:`Dihedral14CFEnergy`.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        nb14_numbers (int32): the number of necessary dihedral 1,4 terms m.
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **LJ_type** (Tensor) - The Lennard-Jones type of each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **charge** (Tensor) - The charge of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **boxlength_f** (Tensor) - The length of molecular simulation box in 3 dimensions.
          The data type is float32 and the shape is :math:`(3,)`.
        - **a_14** (Tensor) - The first atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **b_14** (Tensor, int32) - [m,], the second atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **lj_scale_factor** (Tensor) - The scale factor for the
          Lennard-Jones part of force correction of each dihedral 1,4 term.
          The data type is float32 and the shape is :math:`(m,)`.
        - **cf_scale_factor** (Tensor) - The scale factor for the
          Coulomb part of force correction for each dihedral 1,4 terms.
          The data type is float and the shape is :math:`(m,)`.
        - **LJ_type_A** (Tensor) - The A parameter in Lennard-Jones scheme of each atom pair type.
          q is the number of atom pair. The data type is float32 and the shape is :math:`(q,)`.
        - **LJ_type_B** (Tensor) - The B parameter in Lennard-Jones shceme of each atom pair type.
          q is the number of atom pair. The data type is float32 and the shape is :math:`(q,)`.

    Outputs:
        - **frc_f** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **atom_energy** (Tensor) - The accumulated potential energy for each atom.
          The data type is float32 and the shape is :math:`(n,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, nb14_numbers, atom_numbers):
        """Initialize Dihedral14LJCFForceWithAtomEnergy."""
        validator.check_value_type('nb14_numbers', nb14_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.dihedral_14_numbers = nb14_numbers
        self.atom_numbers = atom_numbers

        self.init_prim_io_names(
            inputs=['uint_crd_f', 'LJtype', 'charge', 'boxlength_f', 'a_14', 'b_14', 'lj_scale_factor',
                    'cf_scale_factor',
                    'LJ_type_A', 'LJ_type_B'],
            outputs=['frc_f', 'atom_energy'])
        self.add_prim_attr('dihedral_14_numbers', self.dihedral_14_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)

    def infer_shape(self, uint_crd_f_shape, ljtype_shape, charge_shape, boxlength_f_shape, a_14_shape, b_14_shape,
                    lj_scale_factor_shape, cf_scale_factor_shape, lj_type_a_shape, lj_type_b_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.dihedral_14_numbers
        q = lj_type_a_shape[0]
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(ljtype_shape), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge_shape), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(boxlength_f_shape), 1, Rel.EQ, "boxlength_f_dim", cls_name)
        validator.check_int(len(a_14_shape), 1, Rel.EQ, "a_14_dim", cls_name)
        validator.check_int(len(b_14_shape), 1, Rel.EQ, "b_14_dim", cls_name)
        validator.check_int(len(lj_scale_factor_shape), 1, Rel.EQ, "lj_scale_factor_dim", cls_name)
        validator.check_int(len(cf_scale_factor_shape), 1, Rel.EQ, "cf_scale_factor_dim", cls_name)
        validator.check_int(len(lj_type_b_shape), 1, Rel.EQ, "LJ_type_B_dim", cls_name)

        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(ljtype_shape[0], n, Rel.EQ, "LJtype_shape", cls_name)
        validator.check_int(charge_shape[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(boxlength_f_shape[0], 3, Rel.EQ, "boxlength_f_shape", cls_name)
        validator.check_int(lj_type_b_shape[0], q, Rel.EQ, "LJ_type_B_shape", cls_name)
        validator.check_int(a_14_shape[0], m, Rel.EQ, "a_14_shape", cls_name)
        validator.check_int(b_14_shape[0], m, Rel.EQ, "b_14_shape", cls_name)
        validator.check_int(lj_scale_factor_shape[0], m, Rel.EQ, "lj_scale_factor_shape", cls_name)
        validator.check_int(cf_scale_factor_shape[0], m, Rel.EQ, "cf_scale_factor_shape", cls_name)
        return uint_crd_f_shape, charge_shape

    def infer_dtype(self, uint_crd_f_dtype, ljtype_dtype, charge_dtype, boxlength_f_type, a_14_type, b_14_type,
                    lj_scale_factor_type, cf_scale_factor_type, lj_type_a_type, lj_type_b_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('boxlength_f', boxlength_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('a_14', a_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('b_14', b_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('lj_scale_factor', lj_scale_factor_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('cf_scale_factor', cf_scale_factor_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('LJ_type_A', lj_type_a_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('LJ_type_B', lj_type_b_type, [mstype.float32], self.name)

        return charge_dtype, charge_dtype


class Dihedral14LJAtomEnergy(PrimitiveWithInfer):
    """
    Add the potential energy caused by Lennard-Jones energy correction for each
    necessary dihedral 1,4 terms to the total potential energy of each atom.

    The calculation formula is the same as operator Dihedral14LJEnergy().

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        nb14_numbers (int32): the number of necessary dihedral 1,4 terms m.
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **LJ_type** (Tensor) - The Lennard-Jones type of each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **charge** (Tensor) - The charge of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **boxlength_f** (Tensor) - The length of molecular simulation box in 3 dimensions.
          The data type is float32 and the shape is :math:`(3,)`.
        - **a_14** (Tensor) - The first atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **b_14** (Tensor) - The second atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **lj_scale_factor** (Tensor) - The scale factor for the
          Lennard-Jones part of force correction of each dihedral 1,4 term.
          The data type is float32 and the shape is :math:`(m,)`.
        - **cf_scale_factor** (Tensor, float) - [m,], the scale factor for the
          Coulomb part of force correction for each dihedral 1,4 terms.
        - **LJ_type_A** (Tensor, float32) - [q,], the A parameter in Lennard-Jones scheme of each atom pair type.
          q is the number of atom pair.
        - **LJ_type_B** (Tensor, float32) - [q,], the B parameter in Lennard-Jones scheme of each atom pair type.
          q is the number of atom pair.

    Outputs:
        - **ene** (Tensor, float32) - [n,], the accumulated potential energy of each atom.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, nb14_numbers, atom_numbers):
        """Initialize Dihedral14LJAtomEnergy."""
        validator.check_value_type('nb14_numbers', nb14_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.dihedral_14_numbers = nb14_numbers
        self.atom_numbers = atom_numbers

        self.init_prim_io_names(
            inputs=['uint_crd_f', 'LJtype', 'charge', 'boxlength_f', 'a_14', 'b_14', 'lj_scale_factor',
                    'LJ_type_A', 'LJ_type_B'],
            outputs=['ene'])
        self.add_prim_attr('dihedral_14_numbers', self.dihedral_14_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)

    def infer_shape(self, uint_crd_f_shape, ljtype_shape, charge_shape, boxlength_f_shape, a_14_shape, b_14_shape,
                    lj_scale_factor_shape, lj_type_a_shape, lj_type_b_shape):
        cls_name = self.name
        n = self.atom_numbers
        q = lj_type_a_shape[0]
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(ljtype_shape), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge_shape), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(boxlength_f_shape), 1, Rel.EQ, "boxlength_f_dim", cls_name)
        validator.check_int(len(a_14_shape), 1, Rel.EQ, "a_14_dim", cls_name)
        validator.check_int(len(b_14_shape), 1, Rel.EQ, "b_14_dim", cls_name)
        validator.check_int(len(lj_scale_factor_shape), 1, Rel.EQ, "lj_scale_factor_dim", cls_name)
        validator.check_int(len(lj_type_b_shape), 1, Rel.EQ, "LJ_type_B_dim", cls_name)

        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(ljtype_shape[0], n, Rel.EQ, "LJtype_shape", cls_name)
        validator.check_int(charge_shape[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(boxlength_f_shape[0], 3, Rel.EQ, "boxlength_f_shape", cls_name)
        validator.check_int(lj_type_b_shape[0], q, Rel.EQ, "LJ_type_B_shape", cls_name)
        m = self.dihedral_14_numbers
        validator.check_int(a_14_shape[0], m, Rel.EQ, "a_14_shape", cls_name)
        validator.check_int(b_14_shape[0], m, Rel.EQ, "b_14_shape", cls_name)
        validator.check_int(lj_scale_factor_shape[0], m, Rel.EQ, "lj_scale_factor_shape", cls_name)
        return ljtype_shape

    def infer_dtype(self, uint_crd_f_dtype, ljtype_dtype, charge_dtype, boxlength_f_type, a_14_type, b_14_type,
                    lj_scale_factor_type, lj_type_a_type, lj_type_b_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('boxlength_f', boxlength_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('a_14', a_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('b_14', b_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('lj_scale_factor', lj_scale_factor_type, [mstype.float32],
                                           self.name)
        validator.check_tensor_dtype_valid('LJ_type_A', lj_type_a_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('LJ_type_B', lj_type_b_type, [mstype.float32], self.name)

        return lj_type_a_type


class Dihedral14CFEnergy(PrimitiveWithInfer):
    """
    Calculate the Coulumb part of 1,4 dihedral energy correction for
    each necessary dihedral terms on the corresponding atoms.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. math::

        dr = (x_a-x_b, y_a-y_b, z_a-z_b)

    .. math::
        E = k*q_a*q_b/|dr|

    Args:
        nb14_numbers (int32): the number of necessary dihedral 1,4 terms m.
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **LJ_type** (Tensor) - The Lennard-Jones type of each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **charge** (Tensor) - The charge of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **boxlength_f** (Tensor) - The length of molecular simulation box in 3 dimensions.
          The data type is float32 and the shape is :math:`(3,)`.
        - **a_14** (Tensor) - The first atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **b_14** (Tensor) - The second atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **cf_scale_factor** (Tensor) - The scale factor for the
          Coulomb part of force correction for each dihedral 1,4 terms.
          The data type is float and the shape is :math:`(m,)`.

    Outputs:
        - **ene** (Tensor) - The accumulated potential energy of each atom.
          The data type is float32 and the shape is :math:`(m,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, nb14_numbers, atom_numbers):
        """Initialize Dihedral14CFEnergy."""
        validator.check_value_type('nb14_numbers', nb14_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.dihedral_14_numbers = nb14_numbers
        self.atom_numbers = atom_numbers

        self.init_prim_io_names(
            inputs=['uint_crd_f', 'LJtype', 'charge', 'boxlength_f', 'a_14', 'b_14', 'cj_scale_factor'],
            outputs=['ene'])
        self.add_prim_attr('dihedral_14_numbers', self.dihedral_14_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)

    def infer_shape(self, uint_crd_f_shape, ljtype_shape, charge_shape, boxlength_f_shape, a_14_shape, b_14_shape,
                    cf_scale_factor_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(ljtype_shape), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge_shape), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(boxlength_f_shape), 1, Rel.EQ, "boxlength_f_dim", cls_name)
        validator.check_int(len(a_14_shape), 1, Rel.EQ, "a_14_dim", cls_name)
        validator.check_int(len(b_14_shape), 1, Rel.EQ, "b_14_dim", cls_name)
        validator.check_int(len(cf_scale_factor_shape), 1, Rel.EQ, "cf_scale_factor_dim", cls_name)

        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(ljtype_shape[0], n, Rel.EQ, "LJtype_shape", cls_name)
        validator.check_int(charge_shape[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(boxlength_f_shape[0], 3, Rel.EQ, "boxlength_f_shape", cls_name)
        m = self.dihedral_14_numbers
        validator.check_int(a_14_shape[0], m, Rel.EQ, "a_14_shape", cls_name)
        validator.check_int(b_14_shape[0], m, Rel.EQ, "b_14_shape", cls_name)
        validator.check_int(cf_scale_factor_shape[0], m, Rel.EQ, "cf_scale_factor_shape", cls_name)
        return [self.dihedral_14_numbers,]

    def infer_dtype(self, uint_crd_f_dtype, ljtype_dtype, charge_dtype, boxlength_f_type, a_14_type, b_14_type,
                    cf_scale_factor_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('boxlength_f', boxlength_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('a_14', a_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('b_14', b_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('lj_scale_factor', cf_scale_factor_type, [mstype.float32],
                                           self.name)

        return charge_dtype


class Dihedral14CFAtomEnergy(PrimitiveWithInfer):
    """
    Add the potential energy caused by Coulumb energy correction for each
    necessary dihedral 1,4 terms to the total potential energy of each atom.

    The calculation formula is the same as operator :class:`Dihedral14CFEnergy`.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        nb14_numbers (int32): the number of necessary dihedral 1,4 terms m.
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **LJ_type** (Tensor) - The Lennard-Jones type of each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **charge** (Tensor) - The charge of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **boxlength_f** (Tensor) - The length of molecular simulation box in 3 dimensions.
          The data type is float32 and the shape is :math:`(3,)`.
        - **a_14** (Tensor) - The first atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **b_14** (Tensor) - The second atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **cf_scale_factor** (Tensor) - The scale factor for the
          Coulomb part of force correction for each dihedral 1,4 terms.
          The data type is float and the shape is :math:`(m,)`.

    Outputs:
        - **ene** (Tensor) - The accumulated potential energy of each atom.
          The data type is float32 and the shape is :math:`(n,)`


    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, nb14_numbers, atom_numbers):
        """Initialize Dihedral14CFAtomEnergy."""
        validator.check_value_type('nb14_numbers', nb14_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.dihedral_14_numbers = nb14_numbers
        self.atom_numbers = atom_numbers

        self.init_prim_io_names(
            inputs=['uint_crd_f', 'LJtype', 'charge', 'boxlength_f', 'a_14', 'b_14', 'cf_scale_factor'],
            outputs=['ene'])
        self.add_prim_attr('dihedral_14_numbers', self.dihedral_14_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)

    def infer_shape(self, uint_crd_f_shape, ljtype_shape, charge_shape, boxlength_f_shape, a_14_shape, b_14_shape,
                    cf_scale_factor_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(ljtype_shape), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge_shape), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(boxlength_f_shape), 1, Rel.EQ, "boxlength_f_dim", cls_name)
        validator.check_int(len(a_14_shape), 1, Rel.EQ, "a_14_dim", cls_name)
        validator.check_int(len(b_14_shape), 1, Rel.EQ, "b_14_dim", cls_name)
        validator.check_int(len(cf_scale_factor_shape), 1, Rel.EQ, "cf_scale_factor_dim", cls_name)

        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(ljtype_shape[0], n, Rel.EQ, "LJtype_shape", cls_name)
        validator.check_int(charge_shape[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(boxlength_f_shape[0], 3, Rel.EQ, "boxlength_f_shape", cls_name)
        m = self.dihedral_14_numbers
        validator.check_int(a_14_shape[0], m, Rel.EQ, "a_14_shape", cls_name)
        validator.check_int(b_14_shape[0], m, Rel.EQ, "b_14_shape", cls_name)
        validator.check_int(cf_scale_factor_shape[0], m, Rel.EQ, "cf_scale_factor_shape", cls_name)
        return ljtype_shape

    def infer_dtype(self, uint_crd_f_dtype, ljtype_dtype, charge_dtype, boxlength_f_type, a_14_type, b_14_type,
                    cf_scale_factor_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('boxlength_f', boxlength_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('a_14', a_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('b_14', b_14_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('cf_scale_factor', cf_scale_factor_type, [mstype.float32],
                                           self.name)

        return charge_dtype


class PMEReciprocalForce(PrimitiveWithInfer):
    """
    Calculate the reciprocal part of long-range Coulumb force using
    PME(Particle Meshed Ewald) method. Assume the number of atoms is n.

    The detailed calculation formula of PME(Particle Meshed Ewald) method
    can be found in this paper: A Smooth Particle Mesh Ewald Method. DOI:
    10.1063/1.470117.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        atom_numbers(int32): the number of atoms, n.
        beta(float32): the PME beta parameter, determined by the
                       non-bond cutoff value and simulation precision tolerance.
        fftx(int32): the number of points for Fourier transform in dimension X.
        ffty(int32): the number of points for Fourier transform in dimension Y.
        fftz(int32): the number of points for Fourier transform in dimension Z.
        box_length_0(float32): the value of boxlength idx 0
        box_length_1(float32): the value of boxlength idx 1
        box_length_2(float32): the value of boxlength idx 2

    Inputs:
        - **uint_crd** (Tensor) - [n, 3], the unsigned int coordinates value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`
        - **charge** (Tensor) - [n,], the charge carried by each atom.
          The data type is float32 and the shape is :math:`(n,)`

    Outputs:
        - **force** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, beta, fftx, ffty, fftz, box_length_0, box_length_1, box_length_2):
        """Initialize PMEReciprocalForce."""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('beta', beta, float, self.name)
        validator.check_value_type('fftx', fftx, int, self.name)
        validator.check_value_type('ffty', ffty, int, self.name)
        validator.check_value_type('fftz', fftz, int, self.name)
        validator.check_value_type('box_length_0', box_length_0, float, self.name)
        validator.check_value_type('box_length_1', box_length_1, float, self.name)
        validator.check_value_type('box_length_2', box_length_2, float, self.name)
        self.atom_numbers = atom_numbers
        self.beta = beta
        self.fftx = fftx
        self.ffty = ffty
        self.fftz = fftz
        self.box_length_0 = box_length_0
        self.box_length_1 = box_length_1
        self.box_length_2 = box_length_2

        self.init_prim_io_names(inputs=['boxlength', 'uint_crd', 'charge'],
                                outputs=['force'])
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('beta', self.beta)
        self.add_prim_attr('fftx', self.fftx)
        self.add_prim_attr('ffty', self.ffty)
        self.add_prim_attr('fftz', self.fftz)
        self.add_prim_attr('box_length_0', self.box_length_0)
        self.add_prim_attr('box_length_1', self.box_length_1)
        self.add_prim_attr('box_length_2', self.box_length_2)

    def infer_shape(self, uint_crd_shape, charge_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_shape), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(charge_shape), 1, Rel.EQ, "charge_dim", cls_name)

        validator.check_int(uint_crd_shape[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd_shape[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(charge_shape[0], n, Rel.EQ, "charge_shape", cls_name)
        return uint_crd_shape

    def infer_dtype(self, uint_crd_type, charge_type):
        validator.check_tensor_dtype_valid('uint_crd', uint_crd_type, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('charge', charge_type, [mstype.float32], self.name)
        return charge_type


class PMEExcludedForce(PrimitiveWithInfer):
    """
    Calculate the excluded  part of long-range Coulumb force using
    PME(Particle Meshed Ewald) method. Assume the number of atoms is
    n, and the length of excluded list is E.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        atom_numbers(int32): the number of atoms, n.
        excluded_numbers(int32): the length of excluded list, E.
        beta(float32): the PME beta parameter, determined by the
          non-bond cutoff value and simulation precision tolerance.

    Inputs:
        - **uint_crd** (Tensor) - The unsigned int coordinates value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`
        - **scaler** (Tensor) - The scale factor between real space
          coordinates and its unsigned int value. The data type is float32 and the shape is :math:`(3,)`
        - **charge** (Tensor) - The charge carried by each atom.
          The data type is float32 and the shape is :math:`(n,)`
        - **excluded_list_start** (Tensor) - The start excluded index
          in excluded list for each atom. The data type is int32 and the shape is :math:`(n,)`
        - **excluded_list** (Tensor) - The contiguous join of excluded
          list of each atom. E is the number of excluded atoms. The data type is int32 and the shape is :math:`(E,)`
        - **excluded_atom_numbers** (Tensor) - The number of atom excluded
          in excluded list for each atom. The data type is int32 and the shape is :math:`(n,)`

    Outputs:
        - **force** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, excluded_numbers, beta):
        """Initialize PMEExcludedForce."""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('excluded_numbers', excluded_numbers, int, self.name)
        validator.check_value_type('beta', beta, float, self.name)
        self.atom_numbers = atom_numbers
        self.excluded_numbers = excluded_numbers
        self.beta = beta
        self.init_prim_io_names(
            inputs=['uint_crd', 'sacler', 'charge', 'excluded_list_start', 'excluded_list', 'excluded_atom_numbers'],
            outputs=['force'])
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('excluded_numbers', self.excluded_numbers)
        self.add_prim_attr('beta', self.beta)

    def infer_shape(self, uint_crd_shape, sacler_shape, charge_shape, excluded_list_start_shape, excluded_list_shape,
                    excluded_atom_numbers_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_shape), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(sacler_shape), 1, Rel.EQ, "sacler_dim", cls_name)
        validator.check_int(len(charge_shape), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(excluded_list_start_shape), 1, Rel.EQ, "excluded_list_start_dim", cls_name)
        validator.check_int(len(excluded_atom_numbers_shape), 1, Rel.EQ, "excluded_atom_numbers_dim", cls_name)

        validator.check_int(uint_crd_shape[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd_shape[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(sacler_shape[0], 3, Rel.EQ, "sacler_shape", cls_name)
        validator.check_int(charge_shape[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(excluded_list_start_shape[0], n, Rel.EQ, "excluded_list_start_shape", cls_name)
        validator.check_int(excluded_atom_numbers_shape[0], n, Rel.EQ, "excluded_atom_numbers_shape", cls_name)
        return uint_crd_shape

    def infer_dtype(self, uint_crd_type, sacler_type, charge_type, excluded_list_start_type, excluded_list_type,
                    excluded_atom_numbers_type):
        validator.check_tensor_dtype_valid('sacler', sacler_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('uint_crd', uint_crd_type, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('charge', charge_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('excluded_list_start', excluded_list_start_type, [mstype.int32],
                                           self.name)
        validator.check_tensor_dtype_valid('excluded_list', excluded_list_type, [mstype.int32],
                                           self.name)
        validator.check_tensor_dtype_valid('excluded_atom_numbers', excluded_atom_numbers_type, [mstype.int32],
                                           self.name)
        return charge_type


class PMEEnergy(PrimitiveWithInfer):
    """
    Calculate the Coulumb energy of the system using PME method.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. math::

        E = sum_{ij} q_iq_j/r_{ij}

    Args:
        atom_numbers(int32): the number of atoms, n.
        excluded_numbers(int32): the length of excluded list, E.
        beta(float32): the PME beta parameter, determined by the
                       non-bond cutoff value and simulation precision tolerance.
        fftx(int32): the number of points for Fourier transform in dimension X.
        ffty(int32): the number of points for Fourier transform in dimension Y.
        fftz(int32): the number of points for Fourier transform in dimension Z.
        box_length_0(float32): the value of boxlength idx 0
        box_length_1(float32): the value of boxlength idx 1
        box_length_2(float32): the value of boxlength idx 2


    Inputs:
        - **uint_crd** (Tensor) - The unsigned int coordinates value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`
        - **charge** (Tensor) - The charge carried by each atom.
          The data type is float32 and the shape is :math:`(n,)`
        - **nl_numbers** - (Tensor) - The each atom.
          The data type is int32 and the shape is :math:`(n, 3)`
        - **nl_serial** - (Tensor) - The neighbor list of each atom, the max number is 800.
          The data type is int32 and the shape is :math:`(n, 800)`
        - **scaler** (Tensor) - The scale factor between real space
          coordinates and its unsigned int value. The data type is float32 and the shape is :math:`(3,)`
        - **excluded_list_start** (Tensor) - The start excluded index
          in excluded list for each atom. The data type is int32 and the shape is :math:`(n,)`
        - **excluded_list** (Tensor) - The contiguous join of excluded
          list of each atom. E is the number of excluded atoms. The data type is int32 and the shape is :math:`(E,)`
        - **excluded_atom_numbers** (Tensor) - The number of atom excluded
          in excluded list for each atom. The data type is int32 and the shape is :math:`(n,)`

    Outputs:
        - **reciprocal_ene** (Tensor) - The reciprocal term of PME energy.
          The data type is float32 and the the shape is :math:`(1,)`.
        - **self_ene** (Tensor) - The self term of PME energy.
          The data type is float32 and the the shape is :math:`(1,)`.
        - **direct_ene** (Tensor) - The direct term of PME energy.
          The data type is float32 and the the shape is :math:`(1,)`.
        - **correction_ene** (Tensor) - The correction term of PME energy.
          The data type is float32 and the the shape is :math:`(1,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, excluded_numbers, beta, fftx, ffty, fftz, box_length_0, box_length_1,
                 box_length_2):
        """Initialize PMEEnergy."""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('excluded_numbers', excluded_numbers, int, self.name)
        validator.check_value_type('beta', beta, float, self.name)
        validator.check_value_type('fftx', fftx, int, self.name)
        validator.check_value_type('ffty', ffty, int, self.name)
        validator.check_value_type('fftz', fftz, int, self.name)
        validator.check_value_type('box_length_0', box_length_0, float, self.name)
        validator.check_value_type('box_length_1', box_length_1, float, self.name)
        validator.check_value_type('box_length_2', box_length_2, float, self.name)
        self.atom_numbers = atom_numbers
        self.excluded_numbers = excluded_numbers
        self.beta = beta
        self.fftx = fftx
        self.ffty = ffty
        self.fftz = fftz
        self.box_length_0 = box_length_0
        self.box_length_1 = box_length_1
        self.box_length_2 = box_length_2
        self.init_prim_io_names(
            inputs=['box_length', 'uint_crd', 'charge', 'nl_numbers', 'nl_serial', 'scaler', 'excluded_list_start',
                    'excluded_list', 'excluded_atom_numbers'],
            outputs=['reciprocal_ene', 'self_ene', 'direct_ene', 'correction_ene'])
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('excluded_numbers', self.excluded_numbers)
        self.add_prim_attr('beta', self.beta)
        self.add_prim_attr('fftx', self.fftx)
        self.add_prim_attr('ffty', self.ffty)
        self.add_prim_attr('fftz', self.fftz)
        self.add_prim_attr('box_length_0', self.box_length_0)
        self.add_prim_attr('box_length_1', self.box_length_1)
        self.add_prim_attr('box_length_2', self.box_length_2)

    def infer_shape(self, uint_crd, charge, nl_numbers, nl_serial, scaler, excluded_list_start,
                    excluded_list, excluded_atom_numbers):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(charge), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(nl_numbers), 1, Rel.EQ, "nl_numbers_dim", cls_name)
        validator.check_int(len(nl_serial), 2, Rel.LE, "nl_serial_dim", cls_name)
        validator.check_int(len(excluded_list_start), 1, Rel.EQ, "excluded_list_start_dim", cls_name)
        validator.check_int(len(excluded_atom_numbers), 1, Rel.EQ, "excluded_atom_numbers_dim", cls_name)
        validator.check_int(len(excluded_list), 1, Rel.GE, "excluded_list", cls_name)

        validator.check_int(uint_crd[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(charge[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(nl_numbers[0], n, Rel.EQ, "nl_numbers_shape[0]", cls_name)
        validator.check_int(nl_serial[0], n, Rel.LE, "nl_serial_shape[0]", cls_name)
        validator.check_int(nl_serial[1], 800, Rel.LE, "nl_serial_shape[1]", cls_name)
        validator.check_int(excluded_list_start[0], n, Rel.EQ, "excluded_list_start_shape", cls_name)
        validator.check_int(excluded_atom_numbers[0], n, Rel.EQ, "excluded_atom_numbers_shape", cls_name)
        validator.check_int(excluded_list[0], 0, Rel.GE, "excluded_list_shape", cls_name)
        return (1,), (1,), (1,), (1,)

    def infer_dtype(self, uint_crd, charge, nl_numbers, nl_serial, scaler, excluded_list_start,
                    excluded_list, excluded_atom_numbers):
        validator.check_tensor_dtype_valid('uint_crd', uint_crd, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('charge', charge, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('nl_numbers', nl_numbers, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('nl_serial', nl_serial, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('excluded_list_start', excluded_list_start, [mstype.int32],
                                           self.name)
        validator.check_tensor_dtype_valid('excluded_list', excluded_list, [mstype.int32],
                                           self.name)
        validator.check_tensor_dtype_valid('excluded_atom_numbers', excluded_atom_numbers, [mstype.int32],
                                           self.name)
        return charge, charge, charge, charge


class LJEnergy(PrimitiveWithInfer):
    """
    Calculate the Van der Waals interaction energy described by Lennard-Jones
    potential for each atom. Assume the number of atoms is n, and the number
    of Lennard-Jones types for all atoms is P, which means there will be
    q = P*(P+1)/2 types of possible Lennard-Jones interactions for all kinds
    of atom pairs.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. math::

        dr = (x_a-x_b, y_a-y_b, z_a-z_b)

    .. math::
        E = A/|dr|^{12} - B/|dr|^{6}

    Args:
        atom_numbers(int32): the number of atoms, n.
        cutoff_square(float32): the square value of cutoff.

    Inputs:
        - **uint_crd** (Tensor, uint32) - The unsigned int coordinate value of each atom.
            The data type is uint32 and the shape is :math:`(n, 3)`
        - **LJtype** (Tensor) - The Lennard-Jones type of each atom.
           The data type is int32 and the shape is :math:`(n,)`
        - **charge** (Tensor) - The charge carried by each atom.
           The data type is float32 and the shape is :math:`(n,)`
        - **scaler** (Tensor) - The scale factor between real
          space coordinate and its unsigned int value. The data type is float32 and the shape is :math:`(3,)`
        - **nl_numbers** - (Tensor) - The each atom.
          The data type is int32 and the shape is :math:`(n,)`
        - **nl_serial** - (Tensor) - The neighbor list of each atom, the max number is 800.
          The data type is int32 and the shape is :math:`(n, 800)`.
        - **d_LJ_A** (Tensor) - The Lennard-Jones A coefficient of each kind of atom pair.
          q is the number of atom pair. The data type is float32 and the shape is :math:`(q,)`.
        - **d_LJ_B** (Tensor) - The Lennard-Jones B coefficient of each kind of atom pair.
          q is the number of atom pair. The data type is float32 and the shape is :math:`(q,)`.

    Outputs:
        - **d_LJ_energy_atom** (Tensor) - The Lennard-Jones potential energy of each atom.
           The data type is float32 and the shape is :math:`(n,)`.
        - **d_LJ_energy_sum** (Scalar), the sum of Lennard-Jones potential energy of each atom.
          The data type is float32.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, cutoff_square):
        """Initialize LJEnergy."""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('cutoff_square', cutoff_square, float, self.name)
        self.atom_numbers = atom_numbers
        self.cutoff_square = cutoff_square
        self.init_prim_io_names(
            inputs=['uint_crd', 'LJtype', 'charge', 'scaler', 'nl_numbers', 'nl_serial', 'd_LJ_A', 'd_LJ_B'],
            outputs=['d_LJ_energy_atom'])
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('cutoff_square', self.cutoff_square)

    def infer_shape(self, uint_crd, ljtype, charge, scaler, nl_numbers, nl_serial, d_lj_a, d_lj_b):
        cls_name = self.name
        n = self.atom_numbers
        q = d_lj_a[0]
        validator.check_int(len(uint_crd), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(ljtype), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(scaler), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(nl_numbers), 1, Rel.EQ, "nl_numbers_dim", cls_name)
        validator.check_int(len(nl_serial), 2, Rel.EQ, "nl_serial_dim", cls_name)
        validator.check_int(len(d_lj_b), 1, Rel.EQ, "d_LJ_B_dim", cls_name)

        validator.check_int(uint_crd[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(ljtype[0], n, Rel.EQ, "LJtype_shape", cls_name)
        validator.check_int(charge[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(scaler[0], 3, Rel.EQ, "scaler_shape", cls_name)
        validator.check_int(nl_numbers[0], n, Rel.EQ, "nl_numbers_shape", cls_name)
        validator.check_int(nl_serial[0], n, Rel.EQ, "nl_serial_shape[0]", cls_name)
        validator.check_int(nl_serial[1], 800, Rel.LE, "nl_serial_shape[1]", cls_name)
        validator.check_int(len(d_lj_a), 1, Rel.EQ, "d_LJ_A_dim", cls_name)
        validator.check_int(d_lj_a[0], q, Rel.EQ, "d_LJ_A_shape[0]", cls_name)
        validator.check_int(d_lj_b[0], q, Rel.EQ, "d_LJ_B_shape[0]", cls_name)
        return charge

    def infer_dtype(self, uint_crd, ljtype, charge, scaler, nl_numbers, nl_serial, d_lj_a, d_lj_b):
        validator.check_tensor_dtype_valid('uint_crd', uint_crd, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('nl_numbers', nl_numbers, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('nl_serial', nl_serial, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('d_LJ_A', d_lj_a, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('d_LJ_B', d_lj_b, [mstype.float32], self.name)
        return charge


class LJForce(PrimitiveWithInfer):
    """
    Calculate the Van der Waals interaction force described by Lennard-Jones
    potential energy for each atom.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. math::

        dr = (x_a-x_b, y_a-y_b, z_a-z_b)

    .. math::

        F = (-12*A/|dr|^{14} + 6*B/|dr|^{8}) * dr

    Args:
        atom_numbers(int32): the number of atoms, n.
        cutoff_square(float32): the square value of cutoff.

    Inputs:
        - **uint_crd** (Tensor, uint32) - [n, 3], the unsigned int coordinate value of each atom.
        - **LJtype** (Tensor, int32) - [n,], the Lennard-Jones type of each atom.
        - **charge** (Tensor, float32) - [n,], the charge carried by each atom.
        - **scaler** (Tensor, float32) - [3,], the scale factor between real
          space coordinate and its unsigned int value.
        - **nl_numbers** - (Tensor, int32) - [n,], the each atom.
        - **nl_serial** - (Tensor, int32) - [n, 800], the neighbor list of each atom, the max number is 800.
        - **d_LJ_A** (Tensor, float32) - [q,], the Lennard-Jones A coefficient of each kind of atom pair.
          q is the number of atom pair.
        - **d_LJ_B** (Tensor, float32) - [q,], the Lennard-Jones B coefficient of each kind of atom pair.
          q is the number of atom pair.

    outputs:
        - **frc** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, cutoff_square):
        """Initialize LJForce."""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('cutoff_square', cutoff_square, float, self.name)
        self.atom_numbers = atom_numbers
        self.cutoff_square = cutoff_square
        self.init_prim_io_names(
            inputs=['uint_crd', 'LJtype', 'charge', 'scaler', 'nl_numbers', 'nl_serial', 'd_LJ_A', 'd_LJ_B'],
            outputs=['frc'])
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('cutoff_square', self.cutoff_square)

    def infer_shape(self, uint_crd, ljtype, charge, scaler, nl_numbers, nl_serial, d_lj_a, d_lj_b):
        cls_name = self.name
        n = self.atom_numbers
        q = d_lj_a[0]
        validator.check_int(len(uint_crd), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(ljtype), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(scaler), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(nl_numbers), 1, Rel.EQ, "nl_numbers_dim", cls_name)
        validator.check_int(len(nl_serial), 2, Rel.EQ, "nl_serial_dim", cls_name)
        validator.check_int(len(d_lj_b), 1, Rel.EQ, "d_LJ_B_dim", cls_name)

        validator.check_int(uint_crd[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(ljtype[0], n, Rel.EQ, "LJtype_shape", cls_name)
        validator.check_int(charge[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(scaler[0], 3, Rel.EQ, "scaler_shape", cls_name)
        validator.check_int(nl_numbers[0], n, Rel.EQ, "nl_numbers_shape", cls_name)
        validator.check_int(nl_serial[0], n, Rel.EQ, "nl_serial_shape[0]", cls_name)
        validator.check_int(nl_serial[1], 800, Rel.EQ, "nl_serial_shape[1]", cls_name)
        validator.check_int(len(d_lj_a), 1, Rel.EQ, "d_LJ_A_dim", cls_name)
        validator.check_int(d_lj_a[0], q, Rel.EQ, "d_LJ_A_shape[0]", cls_name)
        validator.check_int(d_lj_b[0], q, Rel.EQ, "d_LJ_B_shape[0]", cls_name)
        return uint_crd

    def infer_dtype(self, uint_crd, ljtype, charge, scaler, nl_numbers, nl_serial, d_lj_a, d_lj_b):
        validator.check_tensor_dtype_valid('uint_crd', uint_crd, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('nl_numbers', nl_numbers, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('nl_serial', nl_serial, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('d_LJ_A', d_lj_a, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('d_LJ_B', d_lj_b, [mstype.float32], self.name)
        return charge


class LJForceWithPMEDirectForce(PrimitiveWithInfer):
    """
    Calculate the Lennard-Jones force and PME direct force together.

    The calculation formula of Lennard-Jones part is the same as operator
    LJForce(), and the PME direct part is within PME method.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        atom_numbers(int32): the number of atoms, n.
        cutoff_square(float32): the square value of cutoff.
        pme_beta(float32): PME beta parameter, same as operator PMEReciprocalForce().

    Inputs:
        - **uint_crd** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **LJtype** (Tensor) - The Lennard-Jones type of each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **charge** (Tensor) - The charge carried by each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **scaler** (Tensor) - The scale factor between real
          space coordinate and its unsigned int value.
          The data type is float32 and the shape is :math:`(3,)`.
        - **nl_numbers** - (Tensor) - The each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **nl_serial** - (Tensor) - The neighbor list of each atom, the max number is 800.
          The data type is int32 and the shape is :math:`(n, 800)`.
        - **d_LJ_A** (Tensor) - The Lennard-Jones A coefficient of each kind of atom pair.
          q is the number of atom pair. The data type is float32 and the shape is :math:`(q,)`.
        - **d_LJ_B** (Tensor) - The Lennard-Jones B coefficient of each kind of atom pair.
          q is the number of atom pair. The data type is float32 and the shape is :math:`(q,)`.

    Outputs:
        - **frc** (Tensor), The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, cutoff, pme_beta):
        """Initialize LJForceWithPMEDirectForce."""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('cutoff', cutoff, float, self.name)
        validator.check_value_type('pme_beta', pme_beta, float, self.name)
        self.atom_numbers = atom_numbers
        self.cutoff = cutoff
        self.pme_beta = pme_beta
        self.init_prim_io_names(
            inputs=['uint_crd', 'LJtype', 'charge', 'scaler', 'nl_numbers', 'nl_serial', 'd_LJ_A', 'd_LJ_B'],
            outputs=['frc'])
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('cutoff', self.cutoff)
        self.add_prim_attr('pme_beta', self.pme_beta)

    def infer_shape(self, uint_crd, ljtype, charge, scaler, nl_numbers, nl_serial, d_lj_a, d_lj_b):
        cls_name = self.name
        n = self.atom_numbers
        q = d_lj_a[0]
        validator.check_int(len(uint_crd), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(ljtype), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(scaler), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(nl_numbers), 1, Rel.EQ, "nl_numbers_dim", cls_name)
        validator.check_int(len(nl_serial), 2, Rel.EQ, "nl_serial_dim", cls_name)
        validator.check_int(len(d_lj_b), 1, Rel.EQ, "d_LJ_B_dim", cls_name)

        validator.check_int(uint_crd[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(ljtype[0], n, Rel.EQ, "LJtype_shape", cls_name)
        validator.check_int(charge[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(scaler[0], 3, Rel.EQ, "scaler_shape", cls_name)
        validator.check_int(nl_numbers[0], n, Rel.EQ, "nl_numbers_shape", cls_name)
        validator.check_int(nl_serial[0], n, Rel.EQ, "nl_serial_shape[0]", cls_name)
        validator.check_int(nl_serial[1], 800, Rel.EQ, "nl_serial_shape[1]", cls_name)
        validator.check_int(len(d_lj_a), 1, Rel.EQ, "d_LJ_A_dim", cls_name)
        validator.check_int(d_lj_a[0], q, Rel.EQ, "d_LJ_A_shape[0]", cls_name)
        validator.check_int(d_lj_b[0], q, Rel.EQ, "d_LJ_B_shape[0]", cls_name)
        return uint_crd

    def infer_dtype(self, uint_crd, ljtype, charge, scaler, nl_numbers, nl_serial, d_lj_a, d_lj_b):
        validator.check_tensor_dtype_valid('uint_crd', uint_crd, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('nl_numbers', nl_numbers, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('nl_serial', nl_serial, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('d_LJ_A', d_lj_a, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('d_LJ_B', d_lj_b, [mstype.float32], self.name)
        return charge


class MDTemperature(PrimitiveWithInfer):
    """
    Compute the MD temperature.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        residue_numbers (int32): the number of residues m.
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **start** (Tensor) - The start atom index of each residue.
          The data type is int32 and the shape is :math:`(m,)`.
        - **end** (Tensor) - The end atom index of each residue.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_vel_f** (Tensor) - The velocity of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **atom_mass** (Tensor) - The mass of each atom.
          The data type is float32 and the shape is :math:`(n,)`.

    Outputs:
        - **ek** (Tensor) - The temperature of each atom.
          The data type is float32 and the shape is :math:`(n,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, residue_numbers, atom_numbers):
        """Initialize MDTemperature."""
        validator.check_value_type('residue_numbers', residue_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.residue_numbers = residue_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('residue_numbers', self.residue_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(
            inputs=['start', 'end', 'atom_vel_f', 'atom_mass'],
            outputs=['ek'])

    def infer_shape(self, start_shape, end_shape, atom_vel_f_shape, atom_mass_shape):
        cls_name = self.name
        n = self.residue_numbers
        m = self.atom_numbers
        validator.check_int(len(start_shape), 1, Rel.EQ, "start", cls_name)
        validator.check_int(start_shape[0], n, Rel.EQ, "end", cls_name)
        validator.check_int(len(end_shape), 1, Rel.EQ, "start", cls_name)
        validator.check_int(end_shape[0], n, Rel.EQ, "end", cls_name)
        validator.check_int(atom_vel_f_shape[0], m, Rel.EQ, "atom_vel_f", cls_name)
        validator.check_int(atom_vel_f_shape[1], 3, Rel.EQ, "atom_vel_f", cls_name)
        validator.check_int(len(atom_mass_shape), 1, Rel.EQ, "atom_mass", cls_name)
        validator.check_int(atom_mass_shape[0], m, Rel.EQ, "atom_mass", cls_name)
        return [n,]

    def infer_dtype(self, start_dtype, end_dtype, atom_vel_f_dtype, atom_mass_dtype):
        validator.check_tensor_dtype_valid('start', start_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('end', end_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_vel_f', atom_vel_f_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_mass', atom_mass_dtype, [mstype.float32], self.name)
        return atom_mass_dtype

class MDIterationLeapFrogWithRF(PrimitiveWithInfer):
    """
    One step of classical leap frog algorithm to solve the finite difference
    Hamiltonian equations of motion for certain system, using Langevin dynamics
    with Liu's thermostat scheme. Assume the number of atoms is n and the target
    control temperature is T.

    Detailed iteration formula can be found in this paper: A unified thermostat
    scheme for efficient configurational sampling for classical/quantum canonical
    ensembles via molecular dynamics. DOI: 10.1063/1.4991621.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Inputs:
        - **float4_numbers** (Scalar) - total length to store random numbers.
          The data type is int32.
        - **atom_numbers** (Scalar) - The number of atoms n.
          The data type is int32.
        - **dt** (Scalar) - time step for finite difference. The data type is float32.
        - **half_dt** (Scalar) - half of time step for finite difference.
          The data type is float32.
        - **exp_gamma** (Scalar) - parameter in Liu's dynamic, equals
          exp(-gamma_ln * dt), where gamma_ln is the firction factor in Langvin
          dynamics. The data type is float32.
        - **max_velocity** (Scalar) - The upper limit of velocity, when the
          veclocity overflows, scale it to the upper limit. The data type is float32.
        - **is_max_velocity** (Scalar) - whether the max velocity control is
          open or not. The data type is int32.
        - **mass_inverse** (Tensor) - The inverse value of
          mass of each atom. The data type is float32 and the shape is :math:`(n,)`.
        - **sqrt_mass** (Tensor) - The inverse square root value
          of effect mass in Liu's dynamics of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **vel** (Tensor) - The velocity of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **frc** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **acc** (Tensor) - The acceleration of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **random force** (Tensor) - The random forces.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Outputs:
        - **res** (Scalar) - The data type is float32.

    Supported Platforms:
        ``GPU``
    Examples:
    """

    @prim_attr_register
    def __init__(self, float4_numbers, atom_numbers, half_dt, dt, exp_gamma, is_max_velocity, max_velocity):
        """Initialize MDIterationLeapFrogWithRF."""
        validator.check_value_type('float4_numbers', float4_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('half_dt', half_dt, float, self.name)
        validator.check_value_type('dt', dt, float, self.name)
        validator.check_value_type('exp_gamma', exp_gamma, float, self.name)
        validator.check_value_type('is_max_velocity', is_max_velocity, int, self.name)
        validator.check_value_type('max_velocity', max_velocity, float, self.name)
        self.float4_numbers = float4_numbers
        self.atom_numbers = atom_numbers
        self.half_dt = half_dt
        self.dt = dt
        self.exp_gamma = exp_gamma
        self.is_max_velocity = is_max_velocity
        self.max_velocity = max_velocity

        self.init_prim_io_names(
            inputs=['mass_inverse', 'sqrt_mass', 'vel_in', 'crd_in', 'frc_in', 'acc_in', 'random_force'],
            outputs=['res'])
        self.add_prim_attr('float4_numbers', self.float4_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('half_dt', self.half_dt)
        self.add_prim_attr('dt', self.dt)
        self.add_prim_attr('exp_gamma', self.exp_gamma)
        self.add_prim_attr('is_max_velocity', self.is_max_velocity)
        self.add_prim_attr('max_velocity', self.max_velocity)

    def infer_shape(self, mass_inverse_shape, sqrt_mass_shape, vel_in_shape, crd_in_shape, frc_in_shape, acc_in_shape,
                    random_force_shape):
        n = self.atom_numbers
        validator.check_int(len(mass_inverse_shape), 1, Rel.EQ, "mass_inverse", self.name)
        validator.check_int(len(sqrt_mass_shape), 1, Rel.EQ, "mass_inverse", self.name)
        validator.check_int(mass_inverse_shape[0], n, Rel.EQ, "mass_inverse", self.name)
        validator.check_int(sqrt_mass_shape[0], n, Rel.EQ, "mass_inverse", self.name)
        validator.check_int(vel_in_shape[0], n, Rel.EQ, "vel_in", self.name)
        validator.check_int(vel_in_shape[1], 3, Rel.EQ, "vel_in", self.name)
        validator.check_int(crd_in_shape[0], n, Rel.EQ, "crd_in", self.name)
        validator.check_int(crd_in_shape[1], 3, Rel.EQ, "crd_in", self.name)
        validator.check_int(frc_in_shape[0], n, Rel.EQ, "frc_in", self.name)
        validator.check_int(frc_in_shape[1], 3, Rel.EQ, "frc_in", self.name)
        validator.check_int(acc_in_shape[0], n, Rel.EQ, "acc_in", self.name)
        validator.check_int(acc_in_shape[1], 3, Rel.EQ, "acc_in", self.name)
        validator.check_int(random_force_shape[0], n, Rel.EQ, "random_force", self.name)
        validator.check_int(random_force_shape[1], 3, Rel.EQ, "random_force", self.name)

        return [1,]

    def infer_dtype(self, mass_inverse_dtype, sqrt_mass_dtype, vel_in_dtype, crd_in_dtype, frc_in_dtype, acc_in_dtype,
                    rf_dtype):
        validator.check_tensor_dtype_valid('mass_inverse', mass_inverse_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('sqrt_mass', sqrt_mass_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('vel_in', vel_in_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('crd_in', crd_in_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('frc_in', frc_in_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('acc_in', acc_in_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('rf', rf_dtype, [mstype.float32], self.name)
        return mstype.float32


class MDIterationLeapFrogLiujian(PrimitiveWithInfer):
    """
    One step of classical leap frog algorithm to solve the finite difference
    Hamiltonian equations of motion for certain system, using Langevin dynamics
    with Liu's thermostat scheme. Assume the number of atoms is n and the target
    control temperature is T.

    Detailed iteration formula can be found in this paper: A unified thermostat
    scheme for efficient configurational sampling for classical/quantum canonical
    ensembles via molecular dynamics. DOI: 10.1063/1.4991621.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        atom_numbers(int32): the number of atoms n.
        dt(float32): time step for finite difference.
        half_dt(float32): half of time step for finite difference.
        exp_gamma(float32): parameter in Liu's dynamic.

    Inputs:
        - **inverse_mass** (Tensor) - The inverse value of
          mass of each atom. The data type is float32 and the shape is :math:`(n)`.
        - **sqrt_mass_inverse** (Tensor) - The inverse square root value
          of effect mass in Liu's dynamics of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **vel** (Tensor) - The velocity of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **frc** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **acc** (Tensor) - The acceleration of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **rand_state** (Tensor) - Random state to generate
          random force. The data type is float32 and the shape is :math:`(math.ceil(atom_numbers * 3.0 / 4.0) * 16, )`.
        - **rand_frc** (Tensor) - The random forces.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Outputs:
        - **output** (Scalar) - The data type is float32.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, half_dt, dt, exp_gamma):
        """Initialize MDIterationLeapFrogLiujian."""
        self.atom_numbers = atom_numbers
        self.half_dt = half_dt
        self.dt = dt
        self.exp_gamma = exp_gamma

        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('half_dt', self.half_dt)
        self.add_prim_attr('dt', self.dt)
        self.add_prim_attr('exp_gamma', self.exp_gamma)
        self.init_prim_io_names(
            inputs=['inverse_mass', 'sqrt_mass_inverse', 'vel', 'crd', 'frc', 'acc', 'rand_state', 'rand_frc'],
            outputs=['output'])

    def infer_shape(self, inverse_mass, sqrt_mass_inverse, vel, crd, frc, acc, rand_state, rand_frc):
        n = self.atom_numbers
        validator.check_int(len(inverse_mass), 1, Rel.EQ, "inverse_mass", self.name)
        validator.check_int(len(sqrt_mass_inverse), 1, Rel.EQ, "sqrt_mass_inverse", self.name)
        validator.check_int(inverse_mass[0], n, Rel.EQ, "inverse_mass", self.name)
        validator.check_int(sqrt_mass_inverse[0], n, Rel.EQ, "sqrt_mass_inverse", self.name)
        return [self.atom_numbers, 3]

    def infer_dtype(self, inverse_mass, sqrt_mass_inverse, vel, crd, frc, acc, rand_state, rand_frc):
        validator.check_tensor_dtype_valid('inverse_mass', inverse_mass, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('sqrt_mass_inverse', sqrt_mass_inverse, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('vel', vel, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('crd', crd, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('frc', frc, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('acc', acc, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('rand_frc', rand_frc, [mstype.float32], self.name)
        return mstype.float32


class CrdToUintCrd(PrimitiveWithInfer):
    """
    Convert FP32 coordinate to Uint32 coordinate.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        atom_numbers(int32): the number of atoms n.

    Inputs:
        - **crd_to_uint_crd_cof** (Tensor) - The scale factor
          between the unsigned int value and the real space coordinates.
          The data type is float32 and the shape is :math:`(3,)`.
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Outputs:
        - **output** (Scalar) - The data type is uint32.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers):
        """Initialize CrdToUintCrd."""
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(
            inputs=['crd_to_uint_crd_cof', 'crd'],
            outputs=['output'])

    def infer_shape(self, crd_to_uint_crd_cof, crd):
        validator.check_int(crd_to_uint_crd_cof[0], 3, Rel.EQ, "crd_to_uint_crd_cof_shape", self.name)
        validator.check_int(crd[0], self.atom_numbers, Rel.EQ, "crd[0]", self.name)
        validator.check_int(crd[1], 3, Rel.EQ, "crd[1]", self.name)
        return crd

    def infer_dtype(self, crd_to_uint_crd_cof, crd):
        validator.check_tensor_dtype_valid('crd_to_uint_crd_cof', crd_to_uint_crd_cof, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('crd', crd, [mstype.float32], self.name)
        return mstype.uint32


class MDIterationSetupRandState(PrimitiveWithInfer):
    """
    Compute the random state of the iteration.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        atom_numbers(int32): the number of atoms n.
        seed(int32): random seed.

    Outputs:
        - **output** (Tensor) random state.
          The data type is float32 and the shape is :math:`(ceil(n * 3 / 4),)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, seed):
        """Initialize MDIterationSetupRandState."""
        self.atom_numbers = atom_numbers
        self.seed = seed
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('seed', self.seed)
        self.init_prim_io_names(
            inputs=[],
            outputs=['output'])

    def infer_shape(self):
        float4_numbers = math.ceil(self.atom_numbers * 3 / 4.0)
        curandsize = 64 / 4
        return [float4_numbers * int(curandsize),]

    def infer_dtype(self):
        return mstype.float32


class TransferCrd(PrimitiveWithInfer):
    """
    Transfer the coordinates to angular and radial.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    Args:
        start_serial(int32): the index start position.
        end_serial(int32): the index end position.
        number(int32): the length of angular and radial.

    Inputs:
        - **crd** (Tensor) - The coordinate of each atom.
          n is the number of atoms. The data type is float32 and the shape is :math:`(n, 3)`.
        - **old_crd** (Tensor) - The last coordinate of each atom.
          n is the number of atoms. The data type is float32 and the shape is :math:`(n, 3)`.
        - **box** (Tensor) - The length of 3 dimensions of the simulation box.
          The data type is float32 and the shape is :math:`(3,)`.

    Outputs:
        - **radial** (Tensor) - The array of radial transferred from coordinates.
          The data type is float32 and the shape is :math:`(number,)`.
        - **angular** (Tensor) - The array of angular transferred from coordinates.
          The data type is float32 and the shape is :math:`(number,)`.
        - **nowarp_crd** (Tensor) - The modified coordinate of each atom for
          computing radial and angular. The data type is float32 and the shape is :math:`(n, 3)`.
        - **box_map_times** (Tensor) - The box map times for radial and  angular.
          The data type is int32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, start_serial, end_serial, number, atom_numbers):
        """Initialize TransferCrd."""
        validator.check_value_type('start_serial', start_serial, int, self.name)
        validator.check_value_type('end_serial', end_serial, int, self.name)
        validator.check_value_type('number', number, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.start_serial = start_serial
        self.end_serial = end_serial
        self.number = number
        self.atom_numbers = atom_numbers
        self.add_prim_attr('start_serial', self.start_serial)
        self.add_prim_attr('end_serial', self.end_serial)
        self.add_prim_attr('number', self.number)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(
            inputs=['crd', 'old_crd', 'box'],
            outputs=['radial', 'angular', 'nowarp_crd', 'box_map_times'])

    def infer_shape(self, crd_shape, old_crd_shape, box_shape):
        n = self.atom_numbers
        validator.check_int(len(crd_shape), 2, Rel.EQ, "crd_dim", self.name)
        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", self.name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", self.name)
        validator.check_int(len(old_crd_shape), 2, Rel.EQ, "old_crd_dim", self.name)
        validator.check_int(old_crd_shape[0], n, Rel.EQ, "old_crd_shape[0]", self.name)
        validator.check_int(old_crd_shape[1], 3, Rel.EQ, "old_crd_shape[1]", self.name)
        validator.check_int(len(box_shape), 1, Rel.EQ, "box_dim", self.name)
        validator.check_int(box_shape[0], 3, Rel.EQ, "box_shape[0]", self.name)
        return [self.number,], [self.number,], [n, 3], [n, 3]

    def infer_dtype(self, crd_dtype, old_crd_dtype, box_dtype):
        validator.check_tensor_dtype_valid('crd', crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('old_crd', old_crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('box', box_dtype, [mstype.float32], self.name)
        return mstype.float32, mstype.float32, mstype.float32, mstype.int32


class FFT3D(PrimitiveWithInfer):
    """
    Forward FFT with Three-Dimensional Input.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **input_tensor** (Tensor, float32) - [fftx, ffty, fftz]

    Outputs:
        - **output_tensor** (complex64) - The real part of the output tensor after
          undergoing fast Fourier transform.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(
            inputs=['input_tensor'],
            outputs=['output_tensor'])

    def infer_shape(self, input_shape):
        self.add_prim_attr('fftx', input_shape[0])
        self.add_prim_attr('ffty', input_shape[1])
        self.add_prim_attr('fftz', input_shape[2])
        return [input_shape[0], input_shape[1], int(input_shape[2]/2)+1]

    def infer_dtype(self, input_dtype):
        validator.check_tensor_dtype_valid('input_tensor', input_dtype, [mstype.float32], self.name)
        return mstype.complex64

class IFFT3D(PrimitiveWithInfer):
    """
    Inverse FFT with Three-Dimensional Input.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **input_tensor** (Tensor, complex64) - [fftx, ffty, fftz]

    Outputs:
        - **output_tensor** (float32) - returns the tensor after undergoing
          inverse Fourier transform.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(
            inputs=['input_tensor'],
            outputs=['output_tensor'])

    def infer_shape(self, input_shape):
        self.add_prim_attr('fftx', input_shape[0])
        self.add_prim_attr('ffty', input_shape[1])
        self.add_prim_attr('fftz', input_shape[2])
        return [input_shape[0], input_shape[1], (input_shape[2]-1)*2]

    def infer_dtype(self, input_dtype):
        validator.check_tensor_dtype_valid('input_tensor', input_dtype, [mstype.complex64], self.name)
        return mstype.float32

class NeighborListUpdate(PrimitiveWithInfer):
    """
    Update (or construct if first time) the Verlet neighbor list for the
    calculation of short-ranged force. Assume the number of atoms is N,
    the number of grids divided is G, the maximum number of atoms in one
    grid is M, the maximum number of atoms in single atom's neighbor list
    is L, and the number of total atom in excluded list is E.

    Args:
        grid_numbers(int32): the total number of grids divided.
        not_first_time(int32): whether to construct the neighbor
          list first time or not.
        nxy(int32): the total number of grids divided in xy plane.
        excluded_atom_numbers(int32): the total atom numbers in the excluded list.
        cutoff(float32): the cutoff distance for short-range force calculation.
        skin(float32): the overflow value of cutoff to maintain a neighbor list.
        cutoff_square(float32): the suqare value of cutoff.
        half_skin_square(float32): skin*skin/4, indicates the maximum
          square value of the distance atom allowed to move between two updates.
        cutoff_with_skin(float32): cutoff + skin, indicates the
          radius of the neighbor list for each atom.
        half_cutoff_with_skin(float32): cutoff_with_skin/2.
        cutoff_with_skin_square(float32): the square value of cutoff_with_skin.
        refresh_interval(int32): the number of iteration steps between two updates of neighbor list.
        max_atom_in_grid_numbers(int32): the maximum number of atoms in one grid.

    Inputs:
        - **atom_numbers_in_grid_bucket** (Tensor, int32) - [G,], the number of atoms in each grid bucket.
        - **bucket** (Tensor, int32) - (Tensor,int32) - [G, M], the atom indices in each grid bucket.
        - **crd** (Tensor, float32) - [N,], the coordinates of each atom.
        - **box_length** (Tensor, float32) - [3,], the length of 3 dimensions of the simulation box.
        - **grid_N** (Tensor, int32) - [3,], the number of grids divided of 3 dimensions of the simulation box.
        - **grid_length_inverse** (float32) - The inverse value of grid length.
        - **atom_in_grid_serial** (Tensor, int32) - [N,], the grid index for each atom.
        - **old_crd** (Tensor, float32) - [N, 3], the coordinates before update of each atom.
        - **crd_to_uint_crd_cof** (Tensor, float32) - [3,], the scale factor
          between the unsigned int value and the real space coordinates.
        - **uint_crd** (Tensor, uint32) - [N, 3], the unsigned int coordinates value fo each atom.
        - **gpointer** (Tensor, int32) - [G, 125], the 125 nearest neighbor grids (including self) of each grid.
          G is the number of nearest neighbor grids.
        - **nl_atom_numbers** (Tensor, int32) - [N,], the number of atoms in neighbor list of each atom.
        - **nl_atom_serial** (Tensor, int32) - [N, L], the indices of atoms in neighbor list of each atom.
        - **uint_dr_to_dr_cof** (Tensor, float32) - [3,], the scale factor between
          the real space coordinates and the unsigned int value.
        - **excluded_list_start** (Tensor, int32) - [N,], the start excluded index in excluded list for each atom.
        - **excluded_numbers** (Tensor, int32) - [N,], the number of atom excluded in excluded list for each atom.
        - **excluded_list** (Tensor, int32) - [E,], the contiguous join of excluded list of each atom.
        - **need_refresh_flag** (Tensor, int32) - [N,], whether the neighbor list of each atom need update or not.
        - **refresh_count** (Tensor, int32) - [1,], count how many iteration steps have passed since last update.

    Outputs:
        - **res** (float32)

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, grid_numbers, atom_numbers, not_first_time, nxy, excluded_atom_numbers,
                 cutoff_square, half_skin_square, cutoff_with_skin, half_cutoff_with_skin, cutoff_with_skin_square,
                 refresh_interval=20, cutoff=10.0, skin=2.0, max_atom_in_grid_numbers=64, max_neighbor_numbers=800):
        self.grid_numbers = grid_numbers
        self.atom_numbers = atom_numbers
        self.refresh_interval = refresh_interval
        self.not_first_time = not_first_time
        self.cutoff = cutoff
        self.skin = skin
        self.max_atom_in_grid_numbers = max_atom_in_grid_numbers
        self.nxy = nxy
        self.excluded_atom_numbers = excluded_atom_numbers
        self.cutoff_square = cutoff_square
        self.half_skin_square = half_skin_square
        self.cutoff_with_skin = cutoff_with_skin
        self.half_cutoff_with_skin = half_cutoff_with_skin
        self.cutoff_with_skin_square = cutoff_with_skin_square
        self.max_neighbor_numbers = max_neighbor_numbers
        self.init_prim_io_names(
            inputs=['atom_numbers_in_grid_bucket', 'bucket', 'crd', 'box_length', 'grid_N', 'grid_length_inverse',
                    'atom_in_grid_serial', 'old_crd', 'crd_to_uint_crd_cof', 'uint_crd', 'gpointer', 'nl_atom_numbers',
                    'nl_atom_serial', 'uint_dr_to_dr_cof', 'excluded_list_start', 'excluded_list', 'excluded_numbers',
                    'need_refresh_flag', 'refresh_count'], outputs=['res'])

        self.add_prim_attr('grid_numbers', self.grid_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('refresh_interval', self.refresh_interval)
        self.add_prim_attr('not_first_time', self.not_first_time)
        self.add_prim_attr('cutoff', self.cutoff)
        self.add_prim_attr('skin', self.skin)
        self.add_prim_attr('max_atom_in_grid_numbers', self.max_atom_in_grid_numbers)
        self.add_prim_attr('nxy', self.nxy)
        self.add_prim_attr('excluded_atom_numbers', self.excluded_atom_numbers)
        self.add_prim_attr('cutoff_square', self.cutoff_square)
        self.add_prim_attr('half_skin_square', self.half_skin_square)
        self.add_prim_attr('cutoff_with_skin', self.cutoff_with_skin)
        self.add_prim_attr('half_cutoff_with_skin', self.half_cutoff_with_skin)
        self.add_prim_attr('cutoff_with_skin_square', self.cutoff_with_skin_square)

    def infer_shape(self, atom_numbers_in_grid_bucket_shape, bucket_shape, crd_shape, box_length_shape, grid_N_shape,
                    grid_length_inverse_shape, atom_in_grid_serial_shape, old_crd_shape, crd_to_uint_crd_cof_shape,
                    uint_crd_shape, gpointer_shape, nl_atom_numbers_shape, nl_atom_serial_shape,
                    uint_dr_to_dr_cof_shape, excluded_list_start_shape, excluded_list_shape, excluded_numbers_shape,
                    need_refresh_flag_shape, refresh_count_shape):
        assert len(atom_numbers_in_grid_bucket_shape) == 1
        assert len(bucket_shape) == 2
        assert len(crd_shape) == 2
        assert len(box_length_shape) == 1
        assert len(grid_N_shape) == 1
        assert len(grid_length_inverse_shape) == 1
        assert len(atom_in_grid_serial_shape) == 1
        assert len(old_crd_shape) == 2
        assert len(crd_to_uint_crd_cof_shape) == 1
        assert len(uint_crd_shape) == 2
        assert len(gpointer_shape) == 2
        assert len(nl_atom_numbers_shape) == 1
        assert len(nl_atom_serial_shape) == 2
        assert len(uint_dr_to_dr_cof_shape) == 1
        assert len(excluded_list_start_shape) == 1
        assert len(excluded_list_shape) == 1
        assert len(excluded_numbers_shape) == 1
        assert len(need_refresh_flag_shape) == 1

        validator.check_int(atom_numbers_in_grid_bucket_shape[0], self.grid_numbers, Rel.EQ,
                            "atom_numbers_in_grid_bucket", self.name)
        validator.check_int(bucket_shape[0], self.grid_numbers, Rel.EQ, "bucket", self.name)
        validator.check_int(bucket_shape[1], self.max_atom_in_grid_numbers, Rel.EQ, "bucket", self.name)
        validator.check_int(crd_shape[0], self.atom_numbers, Rel.EQ, "crd", self.name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd", self.name)
        validator.check_int(box_length_shape[0], 3, Rel.EQ, "box_length", self.name)
        validator.check_int(grid_N_shape[0], 3, Rel.EQ, "grid_N", self.name)
        validator.check_int(grid_length_inverse_shape[0], 3, Rel.EQ, "grid_length_inverse", self.name)
        validator.check_int(atom_in_grid_serial_shape[0], self.atom_numbers, Rel.EQ, "atom_in_grid_serial",
                            self.name)
        validator.check_int(old_crd_shape[0], self.atom_numbers, Rel.EQ, "old_crd", self.name)
        validator.check_int(old_crd_shape[1], 3, Rel.EQ, "old_crd", self.name)
        validator.check_int(crd_to_uint_crd_cof_shape[0], 3, Rel.EQ, "crd_to_uint_crd_cof", self.name)
        validator.check_int(uint_crd_shape[0], self.atom_numbers, Rel.EQ, "uint_crd", self.name)
        validator.check_int(uint_crd_shape[1], 3, Rel.EQ, "uint_crd", self.name)
        validator.check_int(gpointer_shape[0], self.grid_numbers, Rel.EQ, "gpointer", self.name)
        validator.check_int(gpointer_shape[1], 125, Rel.EQ, "gpointer", self.name)
        validator.check_int(nl_atom_numbers_shape[0], self.atom_numbers, Rel.EQ, "nl_atom_numbers", self.name)
        validator.check_int(nl_atom_serial_shape[0], self.atom_numbers, Rel.EQ, "nl_atom_serial", self.name)
        validator.check_int(nl_atom_serial_shape[1], self.max_neighbor_numbers, Rel.EQ, "nl_atom_serial",
                            self.name)
        validator.check_int(uint_dr_to_dr_cof_shape[0], 3, Rel.EQ, "uint_dr_to_dr_cof", self.name)
        validator.check_int(excluded_list_start_shape[0], self.atom_numbers, Rel.EQ, "excluded_list_start",
                            self.name)
        validator.check_int(excluded_list_shape[0], self.excluded_atom_numbers, Rel.EQ, "excluded_list",
                            self.name)
        validator.check_int(excluded_numbers_shape[0], self.atom_numbers, Rel.EQ, "excluded_numbers", self.name)
        validator.check_int(need_refresh_flag_shape[0], 1, Rel.EQ, "need_refresh_flag", self.name)

        return [1,]

    def infer_dtype(self, atom_numbers_in_grid_bucket_dtype, bucket_dtype, crd_dtype, box_length_dtype, grid_N_dtype,
                    grid_length_inverse_dtype, atom_in_grid_serial_dtype, old_crd_dtype, crd_to_uint_crd_cof_dtype,
                    uint_crd_dtype, gpointer_dtype, nl_atom_numbers_dtype, nl_atom_serial_dtype,
                    uint_dr_to_dr_cof_dtype, excluded_list_start_dtype, excluded_list_dtype, excluded_numbers_dtype,
                    need_refresh_flag_dtype, refresh_count_dtype):
        validator.check_tensor_dtype_valid('atom_numbers_in_grid_bucket', atom_numbers_in_grid_bucket_dtype,
                                           [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('bucket', bucket_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('crd', crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('box_length', box_length_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('grid_N', grid_N_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('grid_length_inverse', grid_length_inverse_dtype, [mstype.float32],
                                           self.name)
        validator.check_tensor_dtype_valid('atom_in_grid_serial', atom_in_grid_serial_dtype, [mstype.int32],
                                           self.name)
        validator.check_tensor_dtype_valid('old_crd', old_crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('crd_to_uint_crd_cof', crd_to_uint_crd_cof_dtype, [mstype.float32],
                                           self.name)
        validator.check_tensor_dtype_valid('uint_crd', uint_crd_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('gpointer', gpointer_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('nl_atom_numbers', nl_atom_numbers_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('nl_atom_serial', nl_atom_serial_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('uint_dr_to_dr_cof', uint_dr_to_dr_cof_dtype, [mstype.float32],
                                           self.name)
        validator.check_tensor_dtype_valid('excluded_list_start', excluded_list_start_dtype, [mstype.int32],
                                           self.name)
        validator.check_tensor_dtype_valid('excluded_list', excluded_list_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('excluded_numbers', excluded_numbers_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('need_refresh_flag', need_refresh_flag_dtype, [mstype.int32],
                                           self.name)

        return mstype.float32

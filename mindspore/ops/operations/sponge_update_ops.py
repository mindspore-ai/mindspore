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
from ..._checkparam import Rel
from ..._checkparam import Validator as validator
from ...common import dtype as mstype


class v0coordinaterefresh(PrimitiveWithInfer):
    """

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor, uint32 ) - [n, 3], the unsigned int coordinate value of each atom.
        - **scaler_f** (Tensor, float32) - [3,], the 3-D scale factor (x, y, z),
          between the real space float coordinates and the unsigned int coordinates.
        - **atom_a** (Tensor, int32) - [m,], the first atom index of each bond.

    Outputs:
        - **crd_f** (float32 Tensor) - [n, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, virtual_numbers):
        validator.check_value_type('virtual_numbers', virtual_numbers, (int), self.name)
        validator.check_value_type('atom_numbers', atom_numbers, (int), self.name)
        self.virtual_numbers = virtual_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('virtual_numbers', self.virtual_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'v_info_f'],
                                outputs=['crd_f'])

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, v_info_f_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        return [n, 3]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, v_info_f_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('v_info_f', v_info_f_type, [mstype.float32], self.name)
        return mstype.float32


class v1coordinaterefresh(PrimitiveWithInfer):
    """

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor, uint32 ) - [n, 3], the unsigned int coordinate value of each atom.
        - **scaler_f** (Tensor, float32) - [3,], the 3-D scale factor (x, y, z),
          between the real space float coordinates and the unsigned int coordinates.
        - **atom_a** (Tensor, int32) - [m,], the first atom index of each bond.

    Outputs:
        - **crd_f** (float32 Tensor) - [n, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, virtual_numbers):
        validator.check_value_type('virtual_numbers', virtual_numbers, (int), self.name)
        validator.check_value_type('atom_numbers', atom_numbers, (int), self.name)
        self.virtual_numbers = virtual_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('virtual_numbers', self.virtual_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'v_info_f'],
                                outputs=['crd_f'])

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, v_info_f_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        return [n, 3]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, v_info_f_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('v_info_f', v_info_f_type, [mstype.float32], self.name)
        return mstype.float32


class v2coordinaterefresh(PrimitiveWithInfer):
    """

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor, uint32 ) - [n, 3], the unsigned int coordinate value of each atom.
        - **scaler_f** (Tensor, float32) - [3,], the 3-D scale factor (x, y, z),
          between the real space float coordinates and the unsigned int coordinates.
        - **atom_a** (Tensor, int32) - [m,], the first atom index of each bond.

    Outputs:
        - **crd_f** (float32 Tensor) - [n, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, virtual_numbers):
        validator.check_value_type('virtual_numbers', virtual_numbers, (int), self.name)
        validator.check_value_type('atom_numbers', atom_numbers, (int), self.name)
        self.virtual_numbers = virtual_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('virtual_numbers', self.virtual_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'v_info_f'],
                                outputs=['crd_f'])

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, v_info_f_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        return [n, 3]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, v_info_f_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('v_info_f', v_info_f_type, [mstype.float32], self.name)
        return mstype.float32


class v3coordinaterefresh(PrimitiveWithInfer):
    """

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor, uint32 ) - [n, 3], the unsigned int coordinate value of each atom.
        - **scaler_f** (Tensor, float32) - [3,], the 3-D scale factor (x, y, z),
          between the real space float coordinates and the unsigned int coordinates.
        - **atom_a** (Tensor, int32) - [m,], the first atom index of each bond.

    Outputs:
        - **crd_f** (float32 Tensor) - [n, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, virtual_numbers):
        validator.check_value_type('virtual_numbers', virtual_numbers, (int), self.name)
        validator.check_value_type('atom_numbers', atom_numbers, (int), self.name)
        self.virtual_numbers = virtual_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('virtual_numbers', self.virtual_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'v_info_f'],
                                outputs=['crd_f'])

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, v_info_f_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        return [n, 3]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, v_info_f_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('v_info_f', v_info_f_type, [mstype.float32], self.name)
        return mstype.float32


class v0forceredistribute(PrimitiveWithInfer):
    """

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor, uint32 ) - [n, 3], the unsigned int coordinate value of each atom.
        - **atom_a** (Tensor, int32) - [m,], the first atom index of each bond.

    Outputs:
        - **crd_f** (float32 Tensor) - [n, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, virtual_numbers):
        validator.check_value_type('atom_numbers', atom_numbers, (int), self.name)
        validator.check_value_type('virtual_numbers', virtual_numbers, (int), self.name)
        self.virtual_numbers = virtual_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('virtual_numbers', self.virtual_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'v_info_f'],
                                outputs=['frc_f'])

    def infer_shape(self, uint_crd_f_shape, v_info_f_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        return [n, 3]

    def infer_dtype(self, uint_crd_f_dtype, v_info_f_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('v_info_f', v_info_f_type, [mstype.float32], self.name)
        return mstype.float32


class v1forceredistribute(PrimitiveWithInfer):
    """

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor, uint32 ) - [n, 3], the unsigned int coordinate value of each atom.
        - **atom_a** (Tensor, int32) - [m,], the first atom index of each bond.

    Outputs:
        - **crd_f** (float32 Tensor) - [n, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, virtual_numbers):
        validator.check_value_type('atom_numbers', atom_numbers, (int), self.name)
        validator.check_value_type('virtual_numbers', virtual_numbers, (int), self.name)
        self.virtual_numbers = virtual_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('virtual_numbers', self.virtual_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'v_info_f'],
                                outputs=['frc_f'])

    def infer_shape(self, uint_crd_f_shape, v_info_f_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        return [n, 3]

    def infer_dtype(self, uint_crd_f_dtype, v_info_f_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('v_info_f', v_info_f_type, [mstype.float32], self.name)
        return mstype.float32


class v2forceredistribute(PrimitiveWithInfer):
    """

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor, uint32 ) - [n, 3], the unsigned int coordinate value of each atom.
        - **atom_a** (Tensor, int32) - [m,], the first atom index of each bond.

    Outputs:
        - **crd_f** (float32 Tensor) - [n, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers):
        validator.check_value_type('atom_numbers', atom_numbers, (int), self.name)
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(inputs=['virtual_numbers', 'uint_crd_f', 'v_info_f', 'frc_f'],
                                outputs=['res'])

    def infer_shape(self, virtual_numbers, uint_crd_f_shape, v_info_f_shape, frc_f):
        return [1,]

    def infer_dtype(self, virtual_numbers, uint_crd_f_dtype, v_info_f_type, frc_f):
        return mstype.float32


class v3forceredistribute(PrimitiveWithInfer):
    """

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor, uint32 ) - [n, 3], the unsigned int coordinate value of each atom.
        - **scaler_f** (Tensor, float32) - [3,], the 3-D scale factor (x, y, z),
          between the real space float coordinates and the unsigned int coordinates.
        - **atom_a** (Tensor, int32) - [m,], the first atom index of each bond.

    Outputs:
        - **crd_f** (float32 Tensor) - [n, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, virtual_numbers):
        validator.check_value_type('atom_numbers', atom_numbers, (int), self.name)
        validator.check_value_type('virtual_numbers', virtual_numbers, (int), self.name)
        self.virtual_numbers = virtual_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('virtual_numbers', self.virtual_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'v_info_f'],
                                outputs=['frc_f'])

    def infer_shape(self, uint_crd_f_shape, scaler_f_shape, v_info_f_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_f_shape), 2, Rel.EQ, "uint_crd_f_dim", cls_name)
        validator.check_int(len(scaler_f_shape), 1, Rel.EQ, "scaler_f_dim", cls_name)
        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(scaler_f_shape[0], 3, Rel.EQ, "scaler_f_shape", cls_name)
        return [n, 3]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, v_info_f_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('v_info_f', v_info_f_type, [mstype.float32], self.name)
        return mstype.float32


class RestrainForce(PrimitiveWithInfer):
    """
    Calculate the restrain force.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, restrain_numbers, atom_numbers, factor):
        self.restrain_numbers = restrain_numbers
        self.atom_numbers = atom_numbers
        self.factor = factor
        self.add_prim_attr('restrain_numbers', self.restrain_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('factor', self.factor)
        self.init_prim_io_names(
            inputs=['restrain_list', 'uint_crd', 'uint_crd_ref', 'scaler'],
            outputs=['frc'])

    def infer_shape(self, restrain_list_shape, uint_crd_shape, uint_crd_ref_shape, scaler_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.restrain_numbers
        validator.check_int(len(uint_crd_shape), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(scaler_shape), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(uint_crd_ref_shape), 2, Rel.EQ, "atom_auint_crd_ref_dim", cls_name)
        validator.check_int(len(restrain_list_shape), 1, Rel.EQ, "restrain_list_dim", cls_name)

        validator.check_int(uint_crd_shape[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd_shape[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(scaler_shape[0], 3, Rel.EQ, "scaler_shape", cls_name)

        validator.check_int(restrain_list_shape[0], m, Rel.EQ, "restrain_list_shape", cls_name)
        validator.check_int(uint_crd_ref_shape[0], n, Rel.EQ, "uint_crd_ref_shape[0]", cls_name)
        validator.check_int(uint_crd_ref_shape[1], 3, Rel.EQ, "uint_crd_ref_shape[1]", cls_name)
        return [n, 3]

    def infer_dtype(self, restrain_list_dtype, uint_crd_f_dtype, uint_crd_ref_dtype, scaler_f_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('restrain_list', restrain_list_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('uint_crd_ref', uint_crd_ref_dtype, [mstype.int32], self.name)
        return mstype.float32


class restrainenergy(PrimitiveWithInfer):
    """
    Calculate the restrain energy.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, restrain_numbers, atom_numbers, weight):
        self.restrain_numbers = restrain_numbers
        self.atom_numbers = atom_numbers
        self.weight = weight
        self.add_prim_attr('restrain_numbers', self.restrain_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('weight', self.weight)
        self.init_prim_io_names(
            inputs=['restrain_list', 'crd', 'crd_ref', 'boxlength'],
            outputs=['ene'])

    def infer_shape(self, restrain_list_shape, crd_shape, crd_ref_shape, boxlength_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.restrain_numbers
        validator.check_int(len(crd_shape), 2, Rel.EQ, "crd_dim", cls_name)
        validator.check_int(len(boxlength_shape), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(crd_ref_shape), 2, Rel.EQ, "crd_ref_shape_dim", cls_name)
        validator.check_int(len(restrain_list_shape), 1, Rel.EQ, "restrain_list_dim", cls_name)

        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(boxlength_shape[0], 3, Rel.EQ, "boxlength_shape", cls_name)

        validator.check_int(restrain_list_shape[0], m, Rel.EQ, "restrain_list_shape", cls_name)
        validator.check_int(crd_ref_shape[0], n, Rel.EQ, "crd_ref_shape[0]", cls_name)
        validator.check_int(crd_ref_shape[1], 3, Rel.EQ, "crd_ref_shape[1]", cls_name)
        return [n,]

    def infer_dtype(self, restrain_list_dtype, crd_f_dtype, crd_ref_dtype, boxlength_type):
        validator.check_tensor_dtype_valid('crd', crd_f_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('boxlength', boxlength_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('restrain_list', restrain_list_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('crd_ref', crd_ref_dtype, [mstype.float32], self.name)
        return mstype.float32


class restrainforcewithatomenergyandvirial(PrimitiveWithInfer):
    """
    Calculate the restrain force with atom energy and virial.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, restrain_numbers, atom_numbers, weight):
        self.restrain_numbers = restrain_numbers
        self.atom_numbers = atom_numbers
        self.weight = weight
        self.add_prim_attr('restrain_numbers', self.restrain_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('weight', self.weight)
        self.init_prim_io_names(
            inputs=['restrain_list', 'crd', 'crd_ref', 'boxlength'],
            outputs=['atom_ene', 'atom_virial', 'frc'])

    def infer_shape(self, restrain_list_shape, crd_shape, crd_ref_shape, boxlength_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.restrain_numbers
        validator.check_int(len(crd_shape), 2, Rel.EQ, "crd_dim", cls_name)
        validator.check_int(len(boxlength_shape), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(crd_ref_shape), 2, Rel.EQ, "crd_ref_shape_dim", cls_name)
        validator.check_int(len(restrain_list_shape), 1, Rel.EQ, "restrain_list_dim", cls_name)

        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(boxlength_shape[0], 3, Rel.EQ, "boxlength_shape", cls_name)

        validator.check_int(restrain_list_shape[0], m, Rel.EQ, "restrain_list_shape", cls_name)
        validator.check_int(crd_ref_shape[0], n, Rel.EQ, "crd_ref_shape[0]", cls_name)
        validator.check_int(crd_ref_shape[1], 3, Rel.EQ, "crd_ref_shape[1]", cls_name)
        return [n,], [n,], [n, 3]

    def infer_dtype(self, restrain_list_dtype, crd_f_dtype, crd_ref_dtype, boxlength_type):
        validator.check_tensor_dtype_valid('crd', crd_f_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('boxlength', boxlength_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('restrain_list', restrain_list_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('crd_ref', crd_ref_dtype, [mstype.float32], self.name)
        return mstype.float32, mstype.float32, mstype.float32


class refreshuintcrd(PrimitiveWithInfer):
    """
    Calculate the restrain force with atom energy and virial.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, half_exp_gamma_plus_half):
        self.atom_numbers = atom_numbers
        self.half_exp_gamma_plus_half = half_exp_gamma_plus_half
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('half_exp_gamma_plus_half', self.half_exp_gamma_plus_half)
        self.init_prim_io_names(
            inputs=['crd', 'quarter_cof', 'test_frc', 'mass_inverse'],
            outputs=['uint_crd'])

    def infer_shape(self, crd_shape, quarter_cof_shape, test_frc_shape, mass_inverse_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(crd_shape), 2, Rel.EQ, "crd_dim", cls_name)
        # validator.check_int(len(quarter_cof_shape), 1, Rel.EQ, "scaler_dim", cls_name)
        # validator.check_int(len(test_frc_shape), 2, Rel.EQ, "crd_ref_shape_dim", cls_name)
        # validator.check_int(len(restrain_list_shape), 1, Rel.EQ, "restrain_list_dim", cls_name)

        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        # validator.check_int(boxlength_shape[0], 3, Rel.EQ, "boxlength_shape", cls_name)

        # validator.check_int(restrain_list_shape[0], m, Rel.EQ, "restrain_list_shape", cls_name)
        # validator.check_int(crd_ref_shape[0], n, Rel.EQ, "crd_ref_shape[0]", cls_name)
        # validator.check_int(crd_ref_shape[1], 3, Rel.EQ, "crd_ref_shape[1]", cls_name)
        return [n, 3]

    def infer_dtype(self, crd_dtype, quarter_cof_dtype, test_frc_dtype, mass_inverse_dtype):
        validator.check_tensor_dtype_valid('crd', crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('quarter_cof', quarter_cof_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('test_frc', test_frc_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('mass_inverse', mass_inverse_dtype, [mstype.float32], self.name)
        return mstype.uint32


class constrainforcecyclewithvirial(PrimitiveWithInfer):
    """
    Calculate the restrain force with atom energy and virial.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, constrain_pair_numbers):
        self.atom_numbers = atom_numbers
        self.constrain_pair_numbers = constrain_pair_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('constrain_pair_numbers', self.constrain_pair_numbers)
        self.init_prim_io_names(
            inputs=['uint_crd', 'scaler', 'pair_dr', 'atom_i_serials', 'atom_j_serials',
                    'constant_rs', 'constrain_ks'],
            outputs=['test_frc', 'atom_virial'])

    def infer_shape(self, uint_crd_shape, scaler_shape, pair_dr_shape, atom_i_serials_shape,
                    atom_j_serials_shape, constant_rs_shape, constrain_ks_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_shape), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(scaler_shape), 1, Rel.EQ, "scaler_dim", cls_name)

        validator.check_int(uint_crd_shape[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd_shape[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(scaler_shape[0], 3, Rel.EQ, "scaler_shape", cls_name)

        return [n, 3], [n, 1]

    def infer_dtype(self, uint_crd_dtype, scaler_dtype, pair_dr_dtype, atom_i_serials_dtype,
                    atom_j_serials_dtype, constant_rs_dtype, constrain_ks_dtype):
        validator.check_tensor_dtype_valid('uint_crd', uint_crd_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pair_dr', pair_dr_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_i_serials', atom_i_serials_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_j_serials', atom_j_serials_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('constant_rs', constant_rs_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('constrain_ks', constrain_ks_dtype, [mstype.float32], self.name)
        return mstype.float32, mstype.float32


class lastcrdtodr(PrimitiveWithInfer):
    """
    Calculate the restrain force with atom energy and virial.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, constrain_pair_numbers):
        self.constrain_pair_numbers = constrain_pair_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('constrain_pair_numbers', self.constrain_pair_numbers)
        self.init_prim_io_names(
            inputs=['crd', 'quarter_cof', 'uint_dr_to_dr', 'atom_i_serials', 'atom_j_serials',
                    'constant_rs', 'constrain_ks'],
            outputs=['pair_dr'])

    def infer_shape(self, crd_shape, quarter_cof_shape, uint_dr_to_dr_shape, atom_i_serials_shape,
                    atom_j_serials_shape, constant_rs_shape, constrain_ks_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.constrain_pair_numbers
        validator.check_int(len(crd_shape), 2, Rel.EQ, "crd_dim", cls_name)
        validator.check_int(len(quarter_cof_shape), 1, Rel.EQ, "quarter_cof_dim", cls_name)

        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(quarter_cof_shape[0], 3, Rel.EQ, "quarter_cof_shape", cls_name)

        return [m, 3]

    def infer_dtype(self, crd_dtype, quarter_cof_dtype, uint_dr_to_dr_dtype, atom_i_serials_dtype,
                    atom_j_serials_dtype, constant_rs_dtype, constrain_ks_dtype):
        validator.check_tensor_dtype_valid('crd', crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('quarter_cof', quarter_cof_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('uint_dr_to_dr', uint_dr_to_dr_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_i_serials', atom_i_serials_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_j_serials', atom_j_serials_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('constant_rs', constant_rs_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('constrain_ks', constrain_ks_dtype, [mstype.float32], self.name)
        return mstype.float32


class refreshcrdvel(PrimitiveWithInfer):
    """
    Calculate the restrain force with atom energy and virial.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, dt_inverse, dt, exp_gamma, half_exp_gamma_plus_half):
        self.atom_numbers = atom_numbers
        self.dt_inverse = dt_inverse
        self.dt = dt
        self.exp_gamma = exp_gamma
        self.half_exp_gamma_plus_half = half_exp_gamma_plus_half
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('dt_inverse', self.dt_inverse)
        self.add_prim_attr('dt', self.dt)
        self.add_prim_attr('exp_gamma', self.exp_gamma)
        self.add_prim_attr('half_exp_gamma_plus_half', self.half_exp_gamma_plus_half)
        self.init_prim_io_names(
            inputs=['crd', 'vel', 'test_frc', 'mass_inverse'],
            outputs=['res'])

    def infer_shape(self, crd_shape, vel_shape, test_frc_shape, mass_inverse_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(crd_shape), 2, Rel.EQ, "crd_dim", cls_name)
        validator.check_int(len(vel_shape), 2, Rel.EQ, "vel_dim", cls_name)

        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(vel_shape[0], n, Rel.EQ, "vel_shape[0]", cls_name)
        validator.check_int(vel_shape[1], 3, Rel.EQ, "vel_shape[1]", cls_name)
        return [1,]

    def infer_dtype(self, crd_dtype, vel_dtype, test_frc_dtype, mass_inverse_dtype):
        validator.check_tensor_dtype_valid('crd', crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('vel', vel_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('test_frc', test_frc_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('mass_inverse', mass_inverse_dtype, [mstype.float32], self.name)
        return mstype.float32


class calculatenowrapcrd(PrimitiveWithInfer):
    """
    Calculate the restrain force with atom energy and virial.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers):
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(
            inputs=['crd', 'box', 'box_map_times'],
            outputs=['nowrap_crd'])

    def infer_shape(self, crd_shape, box_shape, box_map_times_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(crd_shape), 2, Rel.EQ, "crd_dim", cls_name)

        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        return [n, 3]

    def infer_dtype(self, crd_dtype, box_dtype, box_map_times_dtype):
        validator.check_tensor_dtype_valid('crd', crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('box', box_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('box_map_times', box_map_times_dtype,
                                           [mstype.float32], self.name)
        return mstype.float32


class refreshboxmaptimes(PrimitiveWithInfer):
    """
    Calculate the restrain force with atom energy and virial.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers):
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(
            inputs=['crd', 'old_crd', 'box_length_inverse', 'box_map_times'],
            outputs=['res'])

    def infer_shape(self, crd_shape, old_crd_shape, box_length_inverse_shape, box_map_times_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(crd_shape), 2, Rel.EQ, "crd_dim", cls_name)
        validator.check_int(len(old_crd_shape), 2, Rel.EQ, "old_crd_dim", cls_name)

        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(old_crd_shape[0], n, Rel.EQ, "old_crd_shape[0]", cls_name)
        validator.check_int(old_crd_shape[1], 3, Rel.EQ, "old_crd_shape[1]", cls_name)
        return [1,]

    def infer_dtype(self, crd_dtype, old_crd_dtype, box_length_inverse_dtype, box_map_times_dtype):
        validator.check_tensor_dtype_valid('crd', crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('old_crd', old_crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('box_length_inverse', box_length_inverse_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('box_map_times', box_map_times_dtype,
                                           [mstype.int32], self.name)
        return mstype.float32


class totalc6get(PrimitiveWithInfer):
    """
    Inverse FFT with Three-Dimensional Input.

    Inputs:
        - **input_real** (Tensor, float32) - [fftx, ffty, fftz]
        - **input_imag** (Tensor, float32) - [fftx, ffty, fftz]

    Outputs:
        - **output_tensor** (float32)

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers):
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(
            inputs=['atom_lj_type', 'lj_b'],
            outputs=['factor'])

    def infer_shape(self, atom_lj_type, lj_b):
        return [1,]

    def infer_dtype(self, atom_lj_type, lj_b):
        validator.check_tensor_dtype_valid('atom_lj_type', atom_lj_type, mstype.int32, self.name)
        validator.check_tensor_dtype_valid('lj_b', lj_b, mstype.float32, self.name)
        return mstype.float32


class copyfrctosystemgrad(PrimitiveWithInfer):
    """

    Args:
        atom_numbers(int32): the number of atoms n.
        bond_numbers(int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor, uint32 ) - [n, 3], the unsigned int coordinate value of each atom.
        - **atom_a** (Tensor, int32) - [m,], the first atom index of each bond.

    Outputs:
        - **crd_f** (float32 Tensor) - [n, 3], the force felt by each atom.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers):
        validator.check_value_type('atom_numbers', atom_numbers, (int), self.name)
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(inputs=['atom_serial', 'system_grad', 'frc'],
                                outputs=['res'])

    def infer_shape(self, atom_serial, system_grad, frc):
        return [1,]

    def infer_dtype(self, atom_serial, system_grad, frc):
        return mstype.float32


class CrdToUintCrdQuarter(PrimitiveWithInfer):
    """
    Convert FP32 coordinate to Uint32 coordinate.

    Args:
        atom_numbers(int32): the number of atoms n.

    Inputs:
        - **crd_to_uint_crd_cof** (Tensor, float32) - [3,], the .
        - **crd** (Tensor, float32) - [n, 3], the coordinate of each atom.

    Outputs:
        - **output** (uint32)

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers):
        """Initialize CrdToUintCrdQuarter."""
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


class MDIterationLeapFrogLiujianWithMaxVel(PrimitiveWithInfer):
    """
    One step of classical leap frog algorithm to solve the finite difference
    Hamiltonian equations of motion for certain system, using Langevin dynamics
    with Liu's thermostat scheme. Assume the number of atoms is n and the target
    control temperature is T.

    Detailed iteration formula can be found in this paper: A unified thermostat
    scheme for efficient configurational sampling for classical/quantum canonical
    ensembles via molecular dynamics. DOI: 10.1063/1.4991621.

    Args:
        atom_numbers(int32): the number of atoms n.
        dt(float32): time step for finite difference.
        half_dt(float32): half of time step for finite difference.
        exp_gamma(float32): parameter in Liu's dynamic, equals
        exp(-gamma_ln * dt), where gamma_ln is the firction factor in Langvin
        dynamics.

    Inputs:
        - **inverse_mass** (Tensor, float32) - [n,], the inverse value of
        mass of each atom.
        - **sqrt_mass_inverse** (Tensor, float32) - [n,], the inverse square root value
        of effect mass in Liu's dynamics of each atom.
        - **vel** (Tensor, float32) - [n, 3], the velocity of each atom.
        - **crd** (Tensor, float32) - [n, 3], the coordinate of each atom.
        - **frc** (Tensor, float32) - [n, 3], the force felt by each atom.
        - **acc** (Tensor, float32) - [n, 3], the acceleration of each atom.
        - **rand_state** (Tensor, float32) - [math.ceil(atom_numbers * 3.0 / 4.0) * 16,], random state to generate
        random force.
        - **rand_frc** (Tensor, float32) - [n, 3], the random forces.

    Outputs:
        - **output** (float32)

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, half_dt, dt, exp_gamma, max_vel):
        """Initialize MDIterationLeapFrogLiujian."""
        self.atom_numbers = atom_numbers
        self.half_dt = half_dt
        self.dt = dt
        self.exp_gamma = exp_gamma
        self.max_vel = max_vel

        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('half_dt', self.half_dt)
        self.add_prim_attr('dt', self.dt)
        self.add_prim_attr('exp_gamma', self.exp_gamma)
        self.add_prim_attr('max_vel', self.max_vel)
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


class GetCenterOfMass(PrimitiveWithInfer):
    """
    Get Center Of Geometry.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, residue_numbers):
        """Initialize GetCenterOfGeometry."""
        validator.check_value_type('residue_numbers', residue_numbers, int, self.name)
        self.residue_numbers = residue_numbers
        self.add_prim_attr('residue_numbers', self.residue_numbers)
        self.init_prim_io_names(
            inputs=['start', 'end', 'crd', 'atom_mass', 'residue_mass_inverse'],
            outputs=['center_of_mass'])

    def infer_shape(self, start, end, crd, atom_mass, residue_mass_inverse):
        m = self.residue_numbers
        return [m, 3]

    def infer_dtype(self, start, end, crd, atom_mass, residue_mass_inverse):
        return mstype.float32


class MapCenterOfMass(PrimitiveWithInfer):
    """
    Get Center Of Geometry.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, residue_numbers, scaler):
        """Initialize GetCenterOfGeometry."""
        validator.check_value_type('residue_numbers', residue_numbers, int, self.name)
        validator.check_value_type('scaler', scaler, float, self.name)
        self.residue_numbers = residue_numbers
        self.scaler = scaler
        self.add_prim_attr('residue_numbers', self.residue_numbers)
        self.add_prim_attr('scaler', self.scaler)
        self.init_prim_io_names(
            inputs=['start', 'end', 'center_of_mass', 'box_length',
                    'no_wrap_crd', 'crd'],
            outputs=['res'])

    def infer_shape(self, start, end, center_of_mass, box_length, no_wrap_crd, crd):
        return [1,]

    def infer_dtype(self, start, end, center_of_mass, box_length, no_wrap_crd, crd):
        return mstype.float32


class NeighborListUpdate(PrimitiveWithInfer):
    """
    Update (or construct if first time) the Verlet neighbor list for the
    calculation of short-ranged force. Assume the number of atoms is n,
    the number of grids divided is G, the maximum number of atoms in one
    grid is m, the maximum number of atoms in single atom's neighbor list
    is L, and the number of total atom in excluded list is E.

    Args:
        grid_numbers(int32): the total number of grids divided.
        not_first_time(int32): whether to construct the neighbor
          list first time or not.
        nxy(int32): the total number of grids divided in xy plane.
        excluded_atom_numbers(int32): the total atom numbers in the excluded list.
        cutoff(float32): the cutoff distance for short-range force calculation. Default: 10.0.
        skin(float32): the overflow value of cutoff to maintain a neighbor list. Default: 2.0.
        cutoff_square(float32): the suqare value of cutoff.
        half_skin_square(float32): skin*skin/4, indicates the maximum
          square value of the distance atom allowed to move between two updates.
        cutoff_with_skin(float32): cutoff + skin, indicates the
          radius of the neighbor list for each atom.
        half_cutoff_with_skin(float32): cutoff_with_skin/2.
        cutoff_with_skin_square(float32): the square value of cutoff_with_skin.
        refresh_interval(int32): the number of iteration steps between two updates of neighbor list. Default: 20.
        max_atom_in_grid_numbers(int32): the maximum number of atoms in one grid. Default: 64.
        max_neighbor_numbers(int32): The maximum number of neighbors. Default: 800.

    Inputs:
        - **atom_numbers_in_grid_bucket** (Tensor, int32) - [G,], the number of atoms in each grid bucket.
        - **bucket** (Tensor, int32) - (Tensor,int32) - [G, m], the atom indices in each grid bucket.
        - **crd** (Tensor, float32) - [n,], the coordinates of each atom.
        - **box_length** (Tensor, float32) - [3,], the length of 3 dimensions of the simulation box.
        - **grid_n** (Tensor, int32) - [3,], the number of grids divided of 3 dimensions of the simulation box.
        - **grid_length_inverse** (float32) - the inverse value of grid length.
        - **atom_in_grid_serial** (Tensor, int32) - [n,], the grid index for each atom.
        - **old_crd** (Tensor, float32) - [n, 3], the coordinates before update of each atom.
        - **crd_to_uint_crd_cof** (Tensor, float32) - [3,], the scale factor
          between the unsigned int value and the real space coordinates.
        - **uint_crd** (Tensor, uint32) - [n, 3], the unsigned int coordinates value fo each atom.
        - **gpointer** (Tensor, int32) - [G, 125], the 125 nearest neighbor grids (including self) of each grid.
          G is the number of nearest neighbor grids.
        - **nl_atom_numbers** (Tensor, int32) - [n,], the number of atoms in neighbor list of each atom.
        - **nl_atom_serial** (Tensor, int32) - [n, L], the indices of atoms in neighbor list of each atom.
        - **uint_dr_to_dr_cof** (Tensor, float32) - [3,], the scale factor between
          the real space coordinates and the unsigned int value.
        - **excluded_list_start** (Tensor, int32) - [n,], the start excluded index in excluded list for each atom.
        - **excluded_numbers** (Tensor, int32) - [n,], the number of atom excluded in excluded list for each atom.
        - **excluded_list** (Tensor, int32) - [E,], the contiguous join of excluded list of each atom.
        - **need_refresh_flag** (Tensor, int32) - [n,], whether the neighbor list of each atom need update or not.
        - **refresh_count** (Tensor, int32) - [1,], count how many iteration steps have passed since last update.

    Outputs:
        - **res** (float32)

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, grid_numbers, atom_numbers, not_first_time, nxy, excluded_atom_numbers,
                 cutoff_square, half_skin_square, cutoff_with_skin, half_cutoff_with_skin, cutoff_with_skin_square,
                 refresh_interval=20, cutoff=10.0, skin=2.0, max_atom_in_grid_numbers=64, max_neighbor_numbers=800,
                 forced_update=0, forced_check=0):
        """Initialize NeighborListUpdate."""
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
        self.forced_update = forced_update
        self.forced_check = forced_check
        self.init_prim_io_names(
            inputs=['atom_numbers_in_grid_bucket', 'bucket', 'crd', 'box_length', 'grid_n', 'grid_length_inverse',
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
        self.add_prim_attr('forced_update', self.forced_update)
        self.add_prim_attr('forced_check', self.forced_check)

    def infer_shape(self, atom_numbers_in_grid_bucket_shape, bucket_shape, crd_shape, box_length_shape, grid_n_shape,
                    grid_length_inverse_shape, atom_in_grid_serial_shape, old_crd_shape, crd_to_uint_crd_cof_shape,
                    uint_crd_shape, gpointer_shape, nl_atom_numbers_shape, nl_atom_serial_shape,
                    uint_dr_to_dr_cof_shape, excluded_list_start_shape, excluded_list_shape, excluded_numbers_shape,
                    need_refresh_flag_shape, refresh_count_shape):
        validator.check_int(len(atom_numbers_in_grid_bucket_shape), 1, Rel.EQ,
                            "atom_numbers_in_grid_bucket_dim", self.name)
        validator.check_int(len(bucket_shape), 2, Rel.EQ, "bucket_dim", self.name)
        validator.check_int(len(crd_shape), 2, Rel.EQ, "crd_dim", self.name)
        validator.check_int(len(box_length_shape), 1, Rel.EQ, "box_length_dim", self.name)
        validator.check_int(len(grid_n_shape), 1, Rel.EQ, "grid_n_dim", self.name)
        validator.check_int(len(grid_length_inverse_shape), 1, Rel.EQ, "grid_length_inverse_dim", self.name)
        validator.check_int(len(atom_in_grid_serial_shape), 1, Rel.EQ, "atom_in_grid_serial_dim", self.name)
        validator.check_int(len(old_crd_shape), 2, Rel.EQ, "old_crd_dim", self.name)
        validator.check_int(len(crd_to_uint_crd_cof_shape), 1, Rel.EQ, "crd_to_uint_crd_cof_dim", self.name)
        validator.check_int(len(uint_crd_shape), 2, Rel.EQ, "uint_crd_dim", self.name)
        validator.check_int(len(gpointer_shape), 2, Rel.EQ, "gpointer_dim", self.name)
        validator.check_int(len(nl_atom_numbers_shape), 1, Rel.EQ, "nl_atom_numbers_dim", self.name)
        validator.check_int(len(nl_atom_serial_shape), 2, Rel.EQ, "nl_atom_serial_dim", self.name)
        validator.check_int(len(uint_dr_to_dr_cof_shape), 1, Rel.EQ, "uint_dr_to_dr_cof_dim", self.name)
        validator.check_int(len(excluded_list_start_shape), 1, Rel.EQ, "excluded_list_start_dim", self.name)
        validator.check_int(len(excluded_list_shape), 1, Rel.EQ, "excluded_list_dim", self.name)
        validator.check_int(len(excluded_numbers_shape), 1, Rel.EQ, "excluded_numbers_dim", self.name)
        validator.check_int(len(need_refresh_flag_shape), 1, Rel.EQ, "need_refresh_flag_dim", self.name)

        validator.check_int(atom_numbers_in_grid_bucket_shape[0], self.grid_numbers, Rel.EQ,
                            "atom_numbers_in_grid_bucket", self.name)
        validator.check_int(bucket_shape[0], self.grid_numbers, Rel.EQ, "bucket", self.name)
        validator.check_int(bucket_shape[1], self.max_atom_in_grid_numbers, Rel.EQ, "bucket", self.name)
        validator.check_int(crd_shape[0], self.atom_numbers, Rel.EQ, "crd", self.name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd", self.name)
        validator.check_int(box_length_shape[0], 3, Rel.EQ, "box_length", self.name)
        validator.check_int(grid_n_shape[0], 3, Rel.EQ, "grid_n", self.name)
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

    def infer_dtype(self, atom_numbers_in_grid_bucket_dtype, bucket_dtype, crd_dtype, box_length_dtype, grid_n_dtype,
                    grid_length_inverse_dtype, atom_in_grid_serial_dtype, old_crd_dtype, crd_to_uint_crd_cof_dtype,
                    uint_crd_dtype, gpointer_dtype, nl_atom_numbers_dtype, nl_atom_serial_dtype,
                    uint_dr_to_dr_cof_dtype, excluded_list_start_dtype, excluded_list_dtype, excluded_numbers_dtype,
                    need_refresh_flag_dtype, refresh_count_dtype):
        validator.check_tensor_dtype_valid('atom_numbers_in_grid_bucket', atom_numbers_in_grid_bucket_dtype,
                                           [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('bucket', bucket_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('crd', crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('box_length', box_length_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('grid_n', grid_n_dtype, [mstype.int32], self.name)
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


class MDIterationLeapFrog(PrimitiveWithInfer):
    """

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, dt):
        """Initialize MDIterationLeapFrogLiujian."""
        self.atom_numbers = atom_numbers
        self.dt = dt

        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('dt', self.dt)
        self.init_prim_io_names(
            inputs=['sqrt_mass_inverse', 'vel', 'crd', 'frc', 'acc', 'inverse_mass'],
            outputs=['res'])

    def infer_shape(self, vel, crd, frc, acc, inverse_mass):
        n = self.atom_numbers
        validator.check_int(len(inverse_mass), 1, Rel.EQ, "inverse_mass", self.name)
        validator.check_int(inverse_mass[0], n, Rel.EQ, "inverse_mass", self.name)
        return [1,]

    def infer_dtype(self, vel, crd, frc, acc, inverse_mass):
        validator.check_tensor_dtype_valid('inverse_mass', inverse_mass, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('vel', vel, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('crd', crd, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('frc', frc, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('acc', acc, [mstype.float32], self.name)
        return mstype.float32


class MDIterationLeapFrogWithMaxVel(PrimitiveWithInfer):
    """

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, dt, max_velocity):
        """Initialize MDIterationLeapFrogLiujian."""
        self.atom_numbers = atom_numbers
        self.dt = dt
        self.max_velocity = max_velocity

        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('dt', self.dt)
        self.add_prim_attr('max_velocity', self.max_velocity)
        self.init_prim_io_names(
            inputs=['sqrt_mass_inverse', 'vel', 'crd', 'frc', 'acc', 'inverse_mass'],
            outputs=['res'])

    def infer_shape(self, vel, crd, frc, acc, inverse_mass):
        n = self.atom_numbers
        validator.check_int(len(inverse_mass), 1, Rel.EQ, "inverse_mass", self.name)
        validator.check_int(inverse_mass[0], n, Rel.EQ, "inverse_mass", self.name)
        return [1,]

    def infer_dtype(self, vel, crd, frc, acc, inverse_mass):
        validator.check_tensor_dtype_valid('inverse_mass', inverse_mass, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('vel', vel, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('crd', crd, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('frc', frc, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('acc', acc, [mstype.float32], self.name)
        return mstype.float32


class MDIterationGradientDescent(PrimitiveWithInfer):
    """

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, learning_rate):
        """Initialize MDIterationGradientDescent."""
        self.atom_numbers = atom_numbers
        self.learning_rate = learning_rate

        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('learning_rate', self.learning_rate)
        self.init_prim_io_names(
            inputs=['crd', 'frc'],
            outputs=['res'])

    def infer_shape(self, crd, frc):
        return [1,]

    def infer_dtype(self, crd, frc):
        validator.check_tensor_dtype_valid('crd', crd, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('frc', frc, [mstype.float32], self.name)
        return mstype.float32


class BondForceWithAtomEnergyAndVirial(PrimitiveWithInfer):
    """
    Calculate bond force and the virial coefficient caused by simple harmonic
    bond for each atom together.

    The calculation formula of the force part is the same as operator BondForce().
    The Virial part is as follows:

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, bond_numbers, atom_numbers):
        """Initialize BondForceWithAtomEnergyAndVirial."""
        validator.check_value_type('bond_numbers', bond_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.bond_numbers = bond_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('bond_numbers', self.bond_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'bond_k', 'bond_r0'],
                                outputs=['frc_f', 'atom_e', 'atom_v'])

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

        return uint_crd_f_shape, [n,], [n,]

    def infer_dtype(self, uint_crd_f_dtype, scaler_f_type, atom_a_type, atom_b_type, bond_k_type, bond_r0_type):
        validator.check_tensor_dtype_valid('uint_crd_f', uint_crd_f_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler_f', scaler_f_type, [mstype.float32], self.name)

        validator.check_tensor_dtype_valid('atom_a', atom_a_type, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_b', atom_b_type, [mstype.int32], self.name)

        validator.check_tensor_dtype_valid('bond_k', bond_k_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('bond_r0', bond_r0_type, [mstype.float32], self.name)
        return mstype.float32, mstype.float32, mstype.float32


class ConstrainForceCycle(PrimitiveWithInfer):
    """
    Calculate the restrain force with atom energy and virial.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, constrain_pair_numbers):
        self.atom_numbers = atom_numbers
        self.constrain_pair_numbers = constrain_pair_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('constrain_pair_numbers', self.constrain_pair_numbers)
        self.init_prim_io_names(
            inputs=['uint_crd', 'scaler', 'pair_dr', 'atom_i_serials', 'atom_j_serials',
                    'constant_rs', 'constrain_ks'],
            outputs=['test_frc'])

    def infer_shape(self, uint_crd_shape, scaler_shape, pair_dr_shape, atom_i_serials_shape,
                    atom_j_serials_shape, constant_rs_shape, constrain_ks_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_shape), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(scaler_shape), 1, Rel.EQ, "scaler_dim", cls_name)

        validator.check_int(uint_crd_shape[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd_shape[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(scaler_shape[0], 3, Rel.EQ, "scaler_shape", cls_name)

        return [n, 3]

    def infer_dtype(self, uint_crd_dtype, scaler_dtype, pair_dr_dtype, atom_i_serials_dtype,
                    atom_j_serials_dtype, constant_rs_dtype, constrain_ks_dtype):
        validator.check_tensor_dtype_valid('uint_crd', uint_crd_dtype, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pair_dr', pair_dr_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_i_serials', atom_i_serials_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_j_serials', atom_j_serials_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('constant_rs', constant_rs_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('constrain_ks', constrain_ks_dtype, [mstype.float32], self.name)
        return mstype.float32

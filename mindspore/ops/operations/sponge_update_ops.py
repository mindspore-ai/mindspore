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
  SPONGE operators for new modules. This is an experimental interface that is subject to change and/or deletion.
"""
import math
from ..primitive import PrimitiveWithInfer, prim_attr_register
from ..._checkparam import Rel
from ..._checkparam import Validator as validator
from ...common import dtype as mstype


class RefreshUintCrd(PrimitiveWithInfer):
    """
    Refresh the unsigned coordinate of each constrained atom in each constrain iteration.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        half_exp_gamma_plus_half (float32): constant value (1.0 + exp(gamma * dt)) if Langvin-Liu thermostat is used,
            where gamma is friction coefficient and dt is the simulation time step, 1.0 otherwise.

    Inputs:
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **quarter_cof** (Tensor) - The 3-D scale factor.
          The data type is float32 and the shape is :math:`(3,)`.
        - **test_frc** (Tensor) - The constraint force.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **mass_inverse** (Tensor) - The inverse value of mass of each atom.
          The data type is float32 and the shape is :math:`(n,)`.

    Outputs:
        - **uint_crd** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, half_exp_gamma_plus_half):
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('half_exp_gamma_plus_half', half_exp_gamma_plus_half, float, self.name)
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
        validator.check_int(len(quarter_cof_shape), 1, Rel.EQ, "quarter_cof_dim", cls_name)
        validator.check_int(len(test_frc_shape), 2, Rel.EQ, "test_frc_dim", cls_name)
        validator.check_int(len(mass_inverse_shape), 1, Rel.EQ, "mass_inverse_dim", cls_name)

        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(quarter_cof_shape[0], 3, Rel.EQ, "quarter_cof_shape", cls_name)
        validator.check_int(test_frc_shape[0], n, Rel.EQ, "test_frc_shape[0]", cls_name)
        validator.check_int(test_frc_shape[1], 3, Rel.EQ, "test_frc_shape[1]", cls_name)
        validator.check_int(mass_inverse_shape[0], n, Rel.EQ, "mass_inverse_shape", cls_name)

        return [n, 3]

    def infer_dtype(self, crd_dtype, quarter_cof_dtype, test_frc_dtype, mass_inverse_dtype):
        validator.check_tensor_dtype_valid('crd', crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('quarter_cof', quarter_cof_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('test_frc', test_frc_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('mass_inverse', mass_inverse_dtype, [mstype.float32], self.name)
        return mstype.uint32


class ConstrainForceCycleWithVirial(PrimitiveWithInfer):
    """
    Calculate the constraint force and virial in each iteration.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        constrain_pair_numbers (int32): the number of constrain pairs m.

    Inputs:
        - **uint_crd** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler** (Tensor) - The 3-D scale factor (x, y, z),
          The data type is float32 and the shape is :math:`(3,)`.
        - **pair_dr** (Tensor) - The displacement vector of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m, 3)`.
        - **atom_i_serials** (Tensor) - The first atom index of each constrained atom pair.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_j_serials** (Tensor) - The second atom index of each constrained atom pair.
          The data type is int32 and the shape is :math:`(m,)`.
        - **constant_rs** (Tensor) - The constrained distance of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m,)`.
        - **constrain_ks** (Tensor) - The coefficient of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **test_frc** (Tensor) - The constraint force.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **atom_virial** (Tensor) - The virial caused by constraint force of each atom.
          The data type is float32 and the shape is :math:`(m,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, constrain_pair_numbers):
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('constrain_pair_numbers', constrain_pair_numbers, int, self.name)
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
        m = self.constrain_pair_numbers
        validator.check_int(len(uint_crd_shape), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(scaler_shape), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(pair_dr_shape), 2, Rel.EQ, "pair_dr_dim", cls_name)
        validator.check_int(len(atom_i_serials_shape), 1, Rel.EQ, "atom_i_serials_dim", cls_name)
        validator.check_int(len(atom_j_serials_shape), 1, Rel.EQ, "atom_j_serials_dim", cls_name)
        validator.check_int(len(constant_rs_shape), 1, Rel.EQ, "constant_rs_dim", cls_name)
        validator.check_int(len(constrain_ks_shape), 1, Rel.EQ, "constrain_ks_dim", cls_name)

        validator.check_int(uint_crd_shape[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd_shape[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(scaler_shape[0], 3, Rel.EQ, "scaler_shape", cls_name)
        validator.check_int(pair_dr_shape[0], m, Rel.EQ, "pair_dr_shape[0]", cls_name)
        validator.check_int(pair_dr_shape[1], 3, Rel.EQ, "pair_dr_shape[1]", cls_name)
        validator.check_int(atom_i_serials_shape[0], m, Rel.EQ, "atom_i_serials_shape", cls_name)
        validator.check_int(atom_j_serials_shape[0], m, Rel.EQ, "atom_j_serials_shape", cls_name)
        validator.check_int(constant_rs_shape[0], m, Rel.EQ, "constant_rs_shape", cls_name)
        validator.check_int(constrain_ks_shape[0], m, Rel.EQ, "constrain_ks_shape", cls_name)
        return [n, 3], [m,]

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


class LastCrdToDr(PrimitiveWithInfer):
    """
    Calculate the diplacement vector of each constrained atom pair.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        constrain_pair_numbers (int32): the number of constrain pairs m.

    Inputs:
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **quarter_cof** (Tensor) - The 3-D scale factor.
          The data type is float32 and the shape is :math:`(3,)`.
        - **uint_dr_to_dr** (Tensor) - The 3-D scale factor (x, y, z)
          The data type is int32 and the shape is :math:`(3,)`..
        - **atom_i_serials** (Tensor) - The first atom index of each constrained atom pair.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_j_serials** (Tensor) - The second atom index of each constrained atom pair.
          The data type is int32 and the shape is :math:`(m,)`.
        - **constant_rs** (Tensor) - The constrained distance of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m,)`.
        - **constrain_ks** (Tensor) - The coefficient of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **pair_dr** (Tensor) - The displacement vector of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, constrain_pair_numbers):
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('constrain_pair_numbers', constrain_pair_numbers, int, self.name)
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
        validator.check_int(len(uint_dr_to_dr_shape), 1, Rel.EQ, "quarter_cof_dim", cls_name)
        validator.check_int(len(atom_i_serials_shape), 1, Rel.EQ, "atom_i_serials_dim", cls_name)
        validator.check_int(len(atom_j_serials_shape), 1, Rel.EQ, "atom_j_serials_dim", cls_name)
        validator.check_int(len(constant_rs_shape), 1, Rel.EQ, "constant_rs_dim", cls_name)
        validator.check_int(len(constrain_ks_shape), 1, Rel.EQ, "constrain_ks_dim", cls_name)

        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(quarter_cof_shape[0], 3, Rel.EQ, "quarter_cof_shape", cls_name)
        validator.check_int(uint_dr_to_dr_shape[0], 3, Rel.EQ, "uint_dr_to_dr_shape", cls_name)
        validator.check_int(atom_i_serials_shape[0], m, Rel.EQ, "atom_i_serials_shape", cls_name)
        validator.check_int(atom_j_serials_shape[0], m, Rel.EQ, "atom_j_serials_shape", cls_name)
        validator.check_int(constant_rs_shape[0], m, Rel.EQ, "constant_rs_shape", cls_name)
        validator.check_int(constrain_ks_shape[0], m, Rel.EQ, "constrain_ks_shape", cls_name)

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


class RefreshCrdVel(PrimitiveWithInfer):
    """
    Refresh the coordinate and velocity of each constrained atom after all iterations have ended.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        dt_inverse (float32): the inverse value of simulation time step.
        dt (float32): the simulation time step.
        exp_gamma (float32): constant value exp(gamma * dt).
        half_exp_gamma_plus_half (float32): constant value (1 + exp_gamma)/2.

    Inputs:
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **vel** (Tensor) - The velocity of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **test_frc** (Tensor) - The constraint force calculated in the last oteration.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **mass_inverse** (Tensor) - The inverse value of mass of each atom.
          The data type is float32 and the shape is :math:`(n,)`.

    Outputs:
        - **res** (Tensor) - The return value after updating successfully.
          The data type is float32 and the shape is :math:`(1,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, dt_inverse, dt, exp_gamma, half_exp_gamma_plus_half):
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('dt', dt, float, self.name)
        validator.check_value_type('dt_inverse', dt_inverse, float, self.name)
        validator.check_value_type('exp_gamma', exp_gamma, float, self.name)
        validator.check_value_type('half_exp_gamma_plus_half', half_exp_gamma_plus_half, float, self.name)
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
        validator.check_int(len(test_frc_shape), 2, Rel.EQ, "test_frc_dim", cls_name)
        validator.check_int(len(mass_inverse_shape), 1, Rel.EQ, "mass_inverse_dim", cls_name)

        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(vel_shape[0], n, Rel.EQ, "vel_shape[0]", cls_name)
        validator.check_int(vel_shape[1], 3, Rel.EQ, "vel_shape[1]", cls_name)
        validator.check_int(test_frc_shape[0], n, Rel.EQ, "test_frc_shape[0]", cls_name)
        validator.check_int(test_frc_shape[1], 3, Rel.EQ, "test_frc_shape[1]", cls_name)
        validator.check_int(mass_inverse_shape[0], n, Rel.EQ, "mass_inverse_shape[0]", cls_name)
        return [1,]

    def infer_dtype(self, crd_dtype, vel_dtype, test_frc_dtype, mass_inverse_dtype):
        validator.check_tensor_dtype_valid('crd', crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('vel', vel_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('test_frc', test_frc_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('mass_inverse', mass_inverse_dtype, [mstype.float32], self.name)
        return mstype.float32


class CalculateNowrapCrd(PrimitiveWithInfer):
    """
    Calculate the inside-box periodic image of each atom.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **box** (Tensor) - The 3-D size of system.
          The data type is float32 and the shape is :math:`(3, )`.
        - **box_map_times** (Tensor) - The number of times each atom has crossed the box.
          The data type is int32 and the shape is :math:`(n, 3)`.

    Outputs:
        - **nowrap_crd** (Tensor) - The inside-box periodic image of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers):
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(
            inputs=['crd', 'box', 'box_map_times'],
            outputs=['nowrap_crd'])

    def infer_shape(self, crd_shape, box_shape, box_map_times_shape):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(crd_shape), 2, Rel.EQ, "crd_dim", cls_name)
        validator.check_int(len(box_shape), 1, Rel.EQ, "box_dim", cls_name)
        validator.check_int(len(box_map_times_shape), 2, Rel.EQ, "box_map_times_dim", cls_name)

        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(box_shape[0], 3, Rel.EQ, "box_shape[0]", cls_name)
        validator.check_int(box_map_times_shape[0], n, Rel.EQ, "box_map_times_shape[0]", cls_name)
        validator.check_int(box_map_times_shape[1], 3, Rel.EQ, "box_map_times_shape[1]", cls_name)
        return [n, 3]

    def infer_dtype(self, crd_dtype, box_dtype, box_map_times_dtype):
        validator.check_tensor_dtype_valid('crd', crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('box', box_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('box_map_times', box_map_times_dtype,
                                           [mstype.float32], self.name)
        return mstype.float32


class RefreshBoxmapTimes(PrimitiveWithInfer):
    """
    Refresh the box-crossing times of each atom.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **old_crd** (Tensor) - The coordinate of each atom at last update.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **box_length_inverse** (Tensor) - The inverse value of box length in 3 dimensions.
          The data type is float32 and the shape is :math:`(3,)`.
        - **box_map_times** (Tensor) - The number of times each atom has crossed the box.
          The data type is int32 and the shape is :math:`(n, 3)`.

    Outputs:
        - **res** (Tensor) - The return value after updating successfully.
          The data type is float32 and the shape is :math:`(1,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers):
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
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
        validator.check_int(len(box_length_inverse_shape), 1, Rel.EQ, "box_length_inverse_dim", cls_name)
        validator.check_int(len(box_map_times_shape), 2, Rel.EQ, "box_map_times_dim", cls_name)

        validator.check_int(crd_shape[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd_shape[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(old_crd_shape[0], n, Rel.EQ, "old_crd_shape[0]", cls_name)
        validator.check_int(old_crd_shape[1], 3, Rel.EQ, "old_crd_shape[1]", cls_name)
        validator.check_int(box_length_inverse_shape[0], 3, Rel.EQ, "box_length_inverse_shape[0]", cls_name)
        validator.check_int(box_map_times_shape[0], n, Rel.EQ, "box_map_times_shape[0]", cls_name)
        validator.check_int(box_map_times_shape[1], 3, Rel.EQ, "box_map_times_shape[1]", cls_name)
        return [1,]

    def infer_dtype(self, crd_dtype, old_crd_dtype, box_length_inverse_dtype, box_map_times_dtype):
        validator.check_tensor_dtype_valid('crd', crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('old_crd', old_crd_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('box_length_inverse', box_length_inverse_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('box_map_times', box_map_times_dtype,
                                           [mstype.int32], self.name)
        return mstype.float32


class Totalc6get(PrimitiveWithInfer):
    """
    Get the average dispersion constant of short range Lennard-Jones interaction,
    for the subsequent long range correction energy and virial. Assume system has m Lennard-Jones types of atoms.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **atom_lj_type** (Tensor) - The Lennard-Jones type of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **lj_b** (Tensor) - The attraction coefficient of each type. the number of pair atoms is m.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **factor** (Tensor) - The average dispersion constant of Lennard-Jones interaction.
          The data type is float32 and the shape is :math:`(1,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers):
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(
            inputs=['atom_lj_type', 'lj_b'],
            outputs=['factor'])

    def infer_shape(self, atom_lj_type, lj_b):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(atom_lj_type), 1, Rel.EQ, "atom_lj_type_dim", cls_name)
        validator.check_int(len(lj_b), 1, Rel.EQ, "LJ_b_dim", cls_name)
        validator.check_int(atom_lj_type[0], n, Rel.EQ, "atom_lj_type_shape[0]", cls_name)
        return [1,]

    def infer_dtype(self, atom_lj_type, lj_b):
        validator.check_tensor_dtype_valid('atom_lj_type', atom_lj_type, mstype.int32, self.name)
        validator.check_tensor_dtype_valid('lj_b', lj_b, mstype.float32, self.name)
        return mstype.float32


class CrdToUintCrdQuarter(PrimitiveWithInfer):
    """
    Convert FP32 coordinate to Uint32 coordinate.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **crd_to_uint_crd_cof** (Tensor) - The crd_to_uint_crd coefficient.
          The data type is float32 and the shape is :math:`(3,)`.
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Outputs:
        - **output** (Tensor) - The unsigned int coordinates.
          The data type is unsigned int32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers):
        """Initialize CrdToUintCrdQuarter"""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.atom_numbers = atom_numbers
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(
            inputs=['crd_to_uint_crd_cof', 'crd'],
            outputs=['output'])

    def infer_shape(self, crd_to_uint_crd_cof, crd):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(crd), 2, Rel.EQ, "crd_dim", cls_name)
        validator.check_int(len(crd_to_uint_crd_cof), 1, Rel.EQ, "crd_to_uint_crd_cof_dim", cls_name)
        validator.check_int(crd_to_uint_crd_cof[0], 3, Rel.EQ, "crd_to_uint_crd_cof_shape", self.name)
        validator.check_int(crd[0], n, Rel.EQ, "crd[0]", self.name)
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
    with Liu's thermostat scheme, but with an maximum velocity limit. Assume the
    number of atoms is n and the target control temperature is T.

    Detailed iteration formula can be found in this paper: A unified thermostat
    scheme for efficient configurational sampling for classical/quantum canonical
    ensembles via molecular dynamics. DOI: 10.1063/1.4991621.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        dt (float32): time step for finite difference.
        half_dt (float32): half of time step for finite difference.
        exp_gamma (float32): parameter in Liu's dynamic, exp(-gamma_ln * dt).
        max_vel (float32): the maximum velocity limit.

    Inputs:
        - **inverse_mass** (Tensor) - The inverse value of mass of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **sqrt_mass_inverse** (Tensor) - The inverse sqrt of the mass in Liu's dynamics of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **vel** (Tensor) - The velocity of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **frc** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **acc** (Tensor) - The acceleration of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **rand_state** (Tensor) - Random state to generate random force.
          The data type is float32 and the shape is :math:`(math.ceil(n * 3.0 / 4.0) * 16, )`.
        - **rand_frc** (Tensor) - The random forces.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Outputs:
        - **output** (float32) - The output coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, half_dt, dt, exp_gamma, max_vel):
        """Initialize MDIterationLeapFrogLiujian"""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('half_dt', half_dt, float, self.name)
        validator.check_value_type('dt', dt, float, self.name)
        validator.check_value_type('exp_gamma', exp_gamma, float, self.name)
        validator.check_value_type('max_vel', max_vel, float, self.name)
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
        validator.check_int(len(inverse_mass), 1, Rel.EQ, "inverse_mass_dim", self.name)
        validator.check_int(len(sqrt_mass_inverse), 1, Rel.EQ, "sqrt_mass_inverse_dim", self.name)
        validator.check_int(len(rand_state), 1, Rel.EQ, "rand_state_dim", self.name)
        validator.check_int(len(rand_frc), 2, Rel.EQ, "rand_frc_dim", self.name)
        validator.check_int(len(vel), 2, Rel.EQ, "vel_dim", self.name)
        validator.check_int(len(crd), 2, Rel.EQ, "crd_dim", self.name)
        validator.check_int(len(frc), 2, Rel.EQ, "frc_dim", self.name)
        validator.check_int(len(acc), 2, Rel.EQ, "acc_dim", self.name)
        validator.check_int(inverse_mass[0], n, Rel.EQ, "inverse_mass", self.name)
        validator.check_int(sqrt_mass_inverse[0], n, Rel.EQ, "sqrt_mass_inverse", self.name)
        validator.check_int(vel[0], n, Rel.EQ, "vel_shape[0]", self.name)
        validator.check_int(vel[1], 3, Rel.EQ, "vel_shape[1]", self.name)
        validator.check_int(crd[0], n, Rel.EQ, "crd_shape[0]", self.name)
        validator.check_int(crd[1], 3, Rel.EQ, "crd_shape[1]", self.name)
        validator.check_int(frc[0], n, Rel.EQ, "frc_shape[0]", self.name)
        validator.check_int(frc[1], 3, Rel.EQ, "frc_shape[1]", self.name)
        validator.check_int(acc[0], n, Rel.EQ, "acc_shape[0]", self.name)
        validator.check_int(acc[1], 3, Rel.EQ, "acc_shape[1]", self.name)
        validator.check_int(rand_frc[0], n, Rel.EQ, "rand_frc_shape[0]", self.name)
        validator.check_int(rand_frc[1], 3, Rel.EQ, "rand_frc_shape[1]", self.name)
        validator.check_int(rand_state[0], math.ceil(self.atom_numbers * 3 / 4.0) * 16, Rel.EQ, "rand_state", self.name)
        return [self.atom_numbers, 3]

    def infer_dtype(self, inverse_mass, sqrt_mass_inverse, vel, crd, frc, acc, rand_state, rand_frc):
        validator.check_tensor_dtype_valid('inverse_mass', inverse_mass, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('sqrt_mass_inverse', sqrt_mass_inverse, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('vel', vel, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('crd', crd, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('frc', frc, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('acc', acc, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('rand_frc', rand_frc, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('rand_state', rand_state, [mstype.float32], self.name)
        return mstype.float32


class GetCenterOfMass(PrimitiveWithInfer):
    """
    Get coordinate of centroid of each residue. Assume system has n atoms.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        residue_numbers (int32): the number of residues m.

    Inputs:
        - **start** (Tensor) - The start atom index of each residue.
          The data type is int32 and the shape is :math:`(m,)`.
        - **end** (Tensor) - The end atom index of each residue.
          The data type is int32 and the shape is :math:`(m,)`.
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **atom_mass** (Tensor) - The mass of each atom and the atom number is n.
          The data type is float32 and the shape is :math:`(n,)`.
        - **residue_mass_inverse** (Tensor) - The inverse of mass of each residue.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **center_of_mass** (Tensor) - The coordinate of centroid of each residue.
          The data type is float32 and the shape is :math:`(m, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, residue_numbers):
        """Initialize GetCenterOfMass"""
        validator.check_value_type('residue_numbers', residue_numbers, int, self.name)
        self.residue_numbers = residue_numbers
        self.add_prim_attr('residue_numbers', self.residue_numbers)
        self.init_prim_io_names(
            inputs=['start', 'end', 'crd', 'atom_mass', 'residue_mass_inverse'],
            outputs=['center_of_mass'])

    def infer_shape(self, start, end, crd, atom_mass, residue_mass_inverse):
        n = crd[0]
        m = self.residue_numbers
        validator.check_int(len(start), 1, Rel.EQ, "start_dim", self.name)
        validator.check_int(len(end), 1, Rel.EQ, "end_dim", self.name)
        validator.check_int(len(crd), 2, Rel.EQ, "crd_dim", self.name)
        validator.check_int(len(atom_mass), 1, Rel.EQ, "atom_mass_dim", self.name)
        validator.check_int(len(residue_mass_inverse), 1, Rel.EQ, "residue_mass_inverse_dim", self.name)
        validator.check_int(start[0], m, Rel.EQ, "start_shape", self.name)
        validator.check_int(end[0], m, Rel.EQ, "end_shape", self.name)
        validator.check_int(crd[0], n, Rel.EQ, "crd_shape[0]", self.name)
        validator.check_int(crd[1], 3, Rel.EQ, "crd_shape[1]", self.name)
        validator.check_int(atom_mass[0], n, Rel.EQ, "atom_mass_shape[0]", self.name)
        validator.check_int(residue_mass_inverse[0], m, Rel.EQ, "residue_mass_inverse_shape", self.name)
        return [m, 3]

    def infer_dtype(self, start, end, crd, atom_mass, residue_mass_inverse):
        validator.check_tensor_dtype_valid('start', start, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('end', end, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('crd', crd, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_mass', atom_mass, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('residue_mass_inverse', residue_mass_inverse, [mstype.float32], self.name)
        return mstype.float32


class MapCenterOfMass(PrimitiveWithInfer):
    """
    Map all atoms in the same residue to the same periodic box, scale if necessary (usually in pressurestat).
    Assume system has n atoms.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        residue_numbers (int32): the number of residues m.

    Inputs:
        - **start** (Tensor) - The start atom index of each residue.
          The data type is int32 and the shape is :math:`(m,)`.
        - **end** (Tensor) - The end atom index of each residue.
          The data type is int32 and the shape is :math:`(m,)`.
        - **center_of_mass** (Tensor) - The coordinate of centroid of each residue.
          The data type is float32 and the shape is :math:`(m, 3)`.
        - **box_length** (Tensor) - The box length of the simulation box.
          The data type is float32 and the shape is :math:`(3,)`.
        - **no_wrap_crd** (Tensor) - The coordinate of each atom before wrap.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **crd** (Tensor) - The coordinate of each atom after wrap.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **scaler** (Tensor) - The scaler of system.
          The data type is float32 and the shape is :math:`(1,)`.

    Outputs:
        - **res** (Tensor) - The return value after updating successfully.
          The data type is float32 and the shape is :math:`(1,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, residue_numbers):
        """Initialize MapCenterOfMass"""
        validator.check_value_type('residue_numbers', residue_numbers, int, self.name)
        self.residue_numbers = residue_numbers
        self.add_prim_attr('residue_numbers', self.residue_numbers)
        self.init_prim_io_names(
            inputs=['start', 'end', 'center_of_mass', 'box_length',
                    'no_wrap_crd', 'crd', 'scaler'],
            outputs=['res'])

    def infer_shape(self, start, end, center_of_mass, box_length, no_wrap_crd, crd, scaler):
        m = self.residue_numbers
        n = crd[0]
        validator.check_int(len(start), 1, Rel.EQ, "start_dim", self.name)
        validator.check_int(len(end), 1, Rel.EQ, "end_dim", self.name)
        validator.check_int(len(center_of_mass), 2, Rel.EQ, "center_of_mass_dim", self.name)
        validator.check_int(len(box_length), 1, Rel.EQ, "box_length_dim", self.name)
        validator.check_int(len(no_wrap_crd), 2, Rel.EQ, "no_wrap_crd_dim", self.name)
        validator.check_int(len(crd), 2, Rel.EQ, "crd_dim", self.name)
        validator.check_int(len(scaler), 1, Rel.EQ, "scaler_dim", self.name)

        validator.check_int(start[0], m, Rel.EQ, "start_shape", self.name)
        validator.check_int(end[0], m, Rel.EQ, "end_shape", self.name)
        validator.check_int(center_of_mass[0], m, Rel.EQ, "center_of_mass_shape[0]", self.name)
        validator.check_int(center_of_mass[1], 3, Rel.EQ, "center_of_mass_shape[1]", self.name)
        validator.check_int(box_length[0], 3, Rel.EQ, "box_length_shape", self.name)
        validator.check_int(scaler[0], 1, Rel.EQ, "scaler_shape", self.name)
        validator.check_int(no_wrap_crd[0], n, Rel.EQ, "no_wrap_crd_shape[0]", self.name)
        validator.check_int(no_wrap_crd[1], 3, Rel.EQ, "no_wrap_crd_shape[1]", self.name)
        validator.check_int(crd[0], n, Rel.EQ, "crd_shape[0]", self.name)
        validator.check_int(crd[1], 3, Rel.EQ, "crd_shape[1]", self.name)
        return [1,]

    def infer_dtype(self, start, end, center_of_mass, box_length, no_wrap_crd, crd, scaler):
        validator.check_tensor_dtype_valid('start', start, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('end', end, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('crd', crd, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('center_of_mass', center_of_mass, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('box_length', box_length, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('no_wrap_crd', no_wrap_crd, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler, [mstype.float32], self.name)
        return mstype.float32


class NeighborListRefresh(PrimitiveWithInfer):
    """
    Update (or construct if first time) the Verlet neighbor list for the
    calculation of short-ranged force.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        grid_numbers (int32): the total number of grids divided G.
        atom_numbers (int32): the number of atoms n.
        not_first_time (int32): whether to construct the neighbor list first time or not.
        nxy (int32): the total number of grids divided in xy plane.
        excluded_atom_numbers (int32): the total atom numbers in the excluded list E.
        cutoff_square (float32): the cutoff square distance for short-range force calculation.
        half_skin_square (float32): the maximum square value of the distance atom allowed to move between two updates.
        cutoff_with_skin (float32): cutoff + skin, indicates the radius of the neighbor list for each atom.
        half_cutoff_with_skin (float32): cutoff_with_skin/2.
        cutoff_with_skin_square (float32): the square value of cutoff_with_skin.
        refresh_interval (int32): the number of iteration steps between two updates of neighbor list. Default: 20.
        cutoff (float32): the cutoff distance for short-range force calculation. Default: 10.0.
        skin (float32): the maximum value of the distance atom allowed to move. Default: 2.0.
        max_atom_in_grid_numbers (int32): the maximum number of atoms in one grid m. Default: 64.
        max_neighbor_numbers (int32): The maximum number of neighbors m. Default: 800.
        forced_update (int32): the flag that decides whether to force an update. Default: 0.
        forced_check (int32): the flag that decides whether to force an check. Default: 0.

    Inputs:
        - **atom_numbers_in_grid_bucket** (Tensor) - The number of atoms in each grid bucket.
          The data type is int32 and the shape is :math:`(G,)`.
        - **bucket** (Tensor) - (Tensor) - The atom indices in each grid bucket.
          The data type is int32 and the shape is :math:`(G, m)`.
        - **crd** (Tensor) - The coordinates of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **box_length** (Tensor) - The box length of the simulation box.
          The data type is float32 and the shape is :math:`(3,)`.
        - **grid_n** (Tensor) - The number of grids divided of 3 dimensions of the simulation box.
          The data type is int32 and the shape is :math:`(3,)`.
        - **grid_length_inverse** (Tensor) - The inverse value of grid length.
          The data type is int32 and the shape is :math:`(3,)`.
        - **atom_in_grid_serial** (Tensor) - The grid index for each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **old_crd** (Tensor) - The coordinates before update of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **crd_to_uint_crd_cof** (Tensor) - The scale factor between the unsigned int coordinate and the real one.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **uint_crd** (Tensor) - The unsigned int coordinates value fo each atom.
          The data type is unsigned int32 and the shape is :math:`(n, 3)`.
        - **gpointer** (Tensor) - The nearest neighbor grids (including self) of each grid.
          The data type is int32 and the shape is :math:`(G, 125)`.
        - **nl_atom_numbers** (Tensor) - The number of atoms in neighbor list of each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **nl_atom_serial** (Tensor) - The indices of atoms in neighbor list of each atom.
          The data type is int32 and the shape is :math:`(n, m)`.
        - **uint_dr_to_dr_cof** (Tensor) - The scale factor.
          The data type is float32 and the shape is :math:`(3,)`.
        - **excluded_list_start** (Tensor) - The start excluded index in excluded list for each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **excluded_list** (Tensor) - The contiguous join of excluded list of each atom.
          The data type is int32 and the shape is :math:`(E,)`.
        - **excluded_numbers** (Tensor) - The number of atom excluded in excluded list for each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **need_refresh_flag** (Tensor) - Whether the neighbor list of each atom need update or not.
          The data type is int32 and the shape is :math:`(1,)`.
        - **refresh_count** (Tensor) - Count how many iteration steps have passed since last update.
          The data type is int32 and the shape is :math:`(1,)`.

    Outputs:
        - **res** (Tensor) - The return value after updating successfully.
          The data type is float32 and the shape is :math:`(1,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, grid_numbers, atom_numbers, not_first_time, nxy, excluded_atom_numbers,
                 cutoff_square, half_skin_square, cutoff_with_skin, half_cutoff_with_skin, cutoff_with_skin_square,
                 refresh_interval=20, cutoff=10.0, skin=2.0, max_atom_in_grid_numbers=64, max_neighbor_numbers=800,
                 forced_update=0, forced_check=0):
        """Initialize NeighborListUpdate"""
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
        validator.check_int(len(refresh_count_shape), 1, Rel.LE, "need_refresh_flag_dim", self.name)
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
        if refresh_count_shape:
            validator.check_int(refresh_count_shape[0], 1, Rel.EQ, "refresh_count_shape", self.name)
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
        validator.check_tensor_dtype_valid('refresh_count', refresh_count_dtype, [mstype.int32],
                                           self.name)

        return mstype.float32


class MDIterationLeapFrog(PrimitiveWithInfer):
    """
    One step of classical leap frog algorithm to solve the finite difference
    Hamiltonian equations of motion for certain system.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        dt (float32): the simulation time step.

    Inputs:
        - **sqrt_mass_inverse** (Tensor) - The square root of the inverse value of the mass of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **vel** (Tensor) - The velocity of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **frc** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **acc** (Tensor) - The acceleration of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **inverse_mass** (Tensor) - The inverse value of mass of each atom.
          The data type is float32 and the shape is :math:`(n,)`.

    Outputs:
        - **res** (Tensor) - The return value after updating successfully.
          The data type is float32 and the shape is :math:`(1,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, dt):
        """Initialize MDIterationLeapFrog"""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('dt', dt, float, self.name)
        self.atom_numbers = atom_numbers
        self.dt = dt
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('dt', self.dt)
        self.init_prim_io_names(
            inputs=['sqrt_mass_inverse', 'vel', 'crd', 'frc', 'acc', 'inverse_mass'],
            outputs=['res'])

    def infer_shape(self, vel, crd, frc, acc, inverse_mass):
        n = self.atom_numbers
        validator.check_int(len(vel), 2, Rel.EQ, "vel_dim", self.name)
        validator.check_int(len(crd), 2, Rel.EQ, "crd_dim", self.name)
        validator.check_int(len(frc), 2, Rel.EQ, "frc_dim", self.name)
        validator.check_int(len(acc), 2, Rel.EQ, "acc_dim", self.name)
        validator.check_int(len(inverse_mass), 1, Rel.EQ, "inverse_mass_dim", self.name)
        validator.check_int(vel[0], n, Rel.EQ, "vel_shape[0]", self.name)
        validator.check_int(vel[1], 3, Rel.EQ, "vel_shape[1]", self.name)
        validator.check_int(crd[0], n, Rel.EQ, "crd_shape[0]", self.name)
        validator.check_int(crd[1], 3, Rel.EQ, "crd_shape[1]", self.name)
        validator.check_int(frc[0], n, Rel.EQ, "frc_shape[0]", self.name)
        validator.check_int(frc[1], 3, Rel.EQ, "frc_shape[1]", self.name)
        validator.check_int(acc[0], n, Rel.EQ, "acc_shape[0]", self.name)
        validator.check_int(acc[1], 3, Rel.EQ, "acc_shape[1]", self.name)
        validator.check_int(inverse_mass[0], n, Rel.EQ, "inverse_mass_shape", self.name)
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
    Leap frog algorithm to solve the Hamiltonian equations of motion with a maximum velocity limit.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        dt (float32): the simulation time step.
        max_velocity (float32): the maximum velocity limit.

    Inputs:
        - **vel** (Tensor) - The velocity of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **frc** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **acc** (Tensor) - The acceleration of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **inverse_mass** (Tensor) - The inverse value of mass of each atom.
          The data type is float32 and the shape is :math:`(n,)`.

    Outputs:
        - **res** (Tensor) - The return value after updating successfully.
          The data type is float32 and the shape is :math:`(1,)`.


    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, dt, max_velocity):
        """Initialize MDIterationLeapFrogWithMaxVel"""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('dt', dt, float, self.name)
        validator.check_value_type('max_velocity', max_velocity, float, self.name)
        self.atom_numbers = atom_numbers
        self.dt = dt
        self.max_velocity = max_velocity

        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('dt', self.dt)
        self.add_prim_attr('max_velocity', self.max_velocity)
        self.init_prim_io_names(
            inputs=['vel', 'crd', 'frc', 'acc', 'inverse_mass'],
            outputs=['res'])

    def infer_shape(self, vel, crd, frc, acc, inverse_mass):
        n = self.atom_numbers
        validator.check_int(len(vel), 2, Rel.EQ, "vel_dim", self.name)
        validator.check_int(len(crd), 2, Rel.EQ, "crd_dim", self.name)
        validator.check_int(len(frc), 2, Rel.EQ, "frc_dim", self.name)
        validator.check_int(len(acc), 2, Rel.EQ, "acc_dim", self.name)
        validator.check_int(len(inverse_mass), 1, Rel.EQ, "inverse_mass_dim", self.name)
        validator.check_int(inverse_mass[0], n, Rel.EQ, "inverse_mass_shape", self.name)
        validator.check_int(vel[0], n, Rel.EQ, "vel_shape[0]", self.name)
        validator.check_int(vel[1], 3, Rel.EQ, "vel_shape[1]", self.name)
        validator.check_int(crd[0], n, Rel.EQ, "crd_shape[0]", self.name)
        validator.check_int(crd[1], 3, Rel.EQ, "crd_shape[1]", self.name)
        validator.check_int(frc[0], n, Rel.EQ, "frc_shape[0]", self.name)
        validator.check_int(frc[1], 3, Rel.EQ, "frc_shape[1]", self.name)
        validator.check_int(acc[0], n, Rel.EQ, "acc_shape[0]", self.name)
        validator.check_int(acc[1], 3, Rel.EQ, "acc_shape[1]", self.name)
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
    Update the coordinate of each atom in the direction of potential for energy minimization.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        learning_rate (float32): the update step length.

    Inputs:
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **frc** (Tensor), The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Output:
        - **res** (Tensor) - The return value after updating successfully.
          The data type is float32 and the shape is :math:`(1,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, learning_rate):
        """Initialize MDIterationGradientDescent"""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('learning_rate', learning_rate, float, self.name)
        self.atom_numbers = atom_numbers
        self.learning_rate = learning_rate
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('learning_rate', self.learning_rate)
        self.init_prim_io_names(
            inputs=['crd', 'frc'],
            outputs=['res'])

    def infer_shape(self, crd, frc):
        n = self.atom_numbers
        validator.check_int(len(crd), 2, Rel.EQ, "crd_dim", self.name)
        validator.check_int(len(frc), 2, Rel.EQ, "frc_dim", self.name)
        validator.check_int(crd[0], n, Rel.EQ, "crd_shape[0]", self.name)
        validator.check_int(crd[1], 3, Rel.EQ, "crd_shape[1]", self.name)
        validator.check_int(frc[0], n, Rel.EQ, "frc_shape[0]", self.name)
        validator.check_int(frc[1], 3, Rel.EQ, "frc_shape[1]", self.name)
        return [1,]

    def infer_dtype(self, crd, frc):
        validator.check_tensor_dtype_valid('crd', crd, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('frc', frc, [mstype.float32], self.name)
        return mstype.float32


class BondForceWithAtomEnergyAndVirial(PrimitiveWithInfer):
    """
    Calculate bond force, harmonic potential energy and atom virial together.

    The calculation formula is the same as operator BondForce() and BondEnergy().

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        bond_numbers (int32): the number of harmonic bonds m.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler_f** (Tensor) - The 3-D scale factor (x, y, z),
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
        - **frc_f** (Tensor) - The force of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **atom_e** (Tensor) - The energy of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **atom_virial** (Tensor) - The virial of each atom.
          The data type is float32 and the shape is :math:`(n,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, bond_numbers, atom_numbers):
        """Initialize BondForceWithAtomEnergyAndVirial"""
        validator.check_value_type('bond_numbers', bond_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.bond_numbers = bond_numbers
        self.atom_numbers = atom_numbers
        self.add_prim_attr('bond_numbers', self.bond_numbers)
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.init_prim_io_names(inputs=['uint_crd_f', 'scaler_f', 'atom_a', 'atom_b', 'bond_k', 'bond_r0'],
                                outputs=['frc_f', 'atom_e', 'atom_virial'])

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
        validator.check_int(atom_a_shape[0], m, Rel.EQ, "atom_a_shape", cls_name)
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


class LJForceWithVirialEnergy(PrimitiveWithInfer):
    """
    Calculate the Lennard-Jones force, virial and atom energy together.

    The calculation formula of Lennard-Jones part is the same as operator
    LJForce(), and the PME direct part is within PME method.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms, n.
        cutoff (float32): the square value of cutoff.
        pme_beta (float32): PME beta parameter, same as operator PMEReciprocalForce().
        max_neighbor_numbers (int32): the max neighbor numbers, default 800.

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
        - **nl_numbers** (Tensor) - The each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **nl_serial** (Tensor) - The neighbor list of each atom, the max number is 800.
          The data type is int32 and the shape is :math:`(n, 800)`.
        - **d_LJ_A** (Tensor) - The Lennard-Jones A coefficient of each kind of atom pair.
          The number of atom pair is q. The data type is float32 and the shape is :math:`(q,)`.
        - **d_LJ_B** (Tensor) - The Lennard-Jones B coefficient of each kind of atom pair.
          The number of atom pair is q. The data type is float32 and the shape is :math:`(q,)`.

    Outputs:
        - **frc** (Tensor), The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **virial** (Tensor), The virial felt by each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **atom_energy** (Tensor), The atom energy felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, cutoff, pme_beta, max_neighbor_numbers=800):
        """Initialize LJForceWithVirialEnergy"""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('cutoff', cutoff, float, self.name)
        validator.check_value_type('pme_beta', pme_beta, float, self.name)
        validator.check_value_type('max_neighbor_numbers', max_neighbor_numbers, int, self.name)
        self.atom_numbers = atom_numbers
        self.cutoff = cutoff
        self.pme_beta = pme_beta
        self.max_neighbor_numbers = max_neighbor_numbers
        self.init_prim_io_names(
            inputs=['uint_crd', 'LJtype', 'charge', 'scaler', 'nl_numbers', 'nl_serial', 'd_LJ_A', 'd_LJ_B'],
            outputs=['frc', 'virial', 'atom_energy'])
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('cutoff', self.cutoff)
        self.add_prim_attr('pme_beta', self.pme_beta)
        self.add_prim_attr('max_neighbor_numbers', self.max_neighbor_numbers)

    def infer_shape(self, uint_crd, ljtype, charge, scaler, nl_numbers, nl_serial, d_lj_a, d_lj_b):
        cls_name = self.name
        n = self.atom_numbers
        q = d_lj_a[0]
        m = self.max_neighbor_numbers
        validator.check_int(len(uint_crd), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(ljtype), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(scaler), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(nl_numbers), 1, Rel.EQ, "nl_numbers_dim", cls_name)
        validator.check_int(len(nl_serial), 2, Rel.EQ, "nl_serial_dim", cls_name)
        validator.check_int(len(d_lj_a), 1, Rel.EQ, "d_LJ_A_dim", cls_name)
        validator.check_int(len(d_lj_b), 1, Rel.EQ, "d_LJ_B_dim", cls_name)

        validator.check_int(uint_crd[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(ljtype[0], n, Rel.EQ, "LJtype_shape", cls_name)
        validator.check_int(charge[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(scaler[0], 3, Rel.EQ, "scaler_shape", cls_name)
        validator.check_int(nl_numbers[0], n, Rel.EQ, "nl_numbers_shape", cls_name)
        validator.check_int(nl_serial[0], n, Rel.EQ, "nl_serial_shape[0]", cls_name)
        validator.check_int(nl_serial[1], m, Rel.EQ, "nl_serial_shape[1]", cls_name)
        validator.check_int(d_lj_a[0], q, Rel.EQ, "d_LJ_A_shape[0]", cls_name)
        validator.check_int(d_lj_b[0], q, Rel.EQ, "d_LJ_B_shape[0]", cls_name)
        return [n, 3], [n,], [n,]

    def infer_dtype(self, uint_crd, ljtype, charge, scaler, nl_numbers, nl_serial, d_lj_a, d_lj_b):
        validator.check_tensor_dtype_valid('uint_crd', uint_crd, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('nl_numbers', nl_numbers, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('nl_serial', nl_serial, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('d_LJ_A', d_lj_a, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('d_LJ_B', d_lj_b, [mstype.float32], self.name)
        return mstype.float32, mstype.float32, mstype.float32


class LJForceWithPMEDirectForceUpdate(PrimitiveWithInfer):
    """
    Calculate the Lennard-Jones force and PME direct force together for pressure.

    The calculation formula of Lennard-Jones part is the same as operator
    LJForce(), and the PME direct part is within PME method.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms, n.
        cutoff (float32): the square value of cutoff.
        pme_beta (float32): PME beta parameter, same as operator PMEReciprocalForce().
        need_update (int32): if need_update = 1, calculate the pressure, default 0.

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
        - **nl_numbers** (Tensor) - The each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **nl_serial** (Tensor) - The neighbor list of each atom, the max number is 800.
          The data type is int32 and the shape is :math:`(n, 800)`.
        - **d_LJ_A** (Tensor) - The Lennard-Jones A coefficient of each kind of atom pair.
          The number of atom pair is q. The data type is float32 and the shape is :math:`(q,)`.
        - **d_LJ_B** (Tensor) - The Lennard-Jones B coefficient of each kind of atom pair.
          The number of atom pair is q. The data type is float32 and the shape is :math:`(q,)`.
        - **beta** (Tensor) - PME beta parameter. The data type is float32 and the shape is :math:`(1,)`.

    Outputs:
        - **frc** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, cutoff, pme_beta, need_update=0):
        """Initialize LJForceWithPMEDirectForce"""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('cutoff', cutoff, float, self.name)
        validator.check_value_type('pme_beta', pme_beta, float, self.name)
        validator.check_value_type('need_update', need_update, int, self.name)
        self.atom_numbers = atom_numbers
        self.cutoff = cutoff
        self.pme_beta = pme_beta
        self.need_update = need_update
        self.init_prim_io_names(
            inputs=['uint_crd', 'LJtype', 'charge', 'scaler', 'nl_numbers', 'nl_serial', 'd_LJ_A', 'd_LJ_B', 'beta'],
            outputs=['frc'])
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('cutoff', self.cutoff)
        self.add_prim_attr('pme_beta', self.pme_beta)
        self.add_prim_attr('need_update', self.need_update)

    def infer_shape(self, uint_crd, ljtype, charge, scaler, nl_numbers, nl_serial, d_lj_a, d_lj_b, beta):
        cls_name = self.name
        n = self.atom_numbers
        q = d_lj_a[0]
        m = nl_serial[1]
        validator.check_int(len(uint_crd), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(ljtype), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(scaler), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(nl_numbers), 1, Rel.EQ, "nl_numbers_dim", cls_name)
        validator.check_int(len(nl_serial), 2, Rel.EQ, "nl_serial_dim", cls_name)
        validator.check_int(len(d_lj_a), 1, Rel.EQ, "d_LJ_A_dim", cls_name)
        validator.check_int(len(d_lj_b), 1, Rel.EQ, "d_LJ_B_dim", cls_name)
        validator.check_int(len(beta), 1, Rel.EQ, "beta_dim", cls_name)

        validator.check_int(uint_crd[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(ljtype[0], n, Rel.EQ, "LJtype_shape", cls_name)
        validator.check_int(charge[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(scaler[0], 3, Rel.EQ, "scaler_shape", cls_name)
        validator.check_int(nl_numbers[0], n, Rel.EQ, "nl_numbers_shape", cls_name)
        validator.check_int(nl_serial[0], n, Rel.EQ, "nl_serial_shape[0]", cls_name)
        validator.check_int(nl_serial[1], m, Rel.EQ, "nl_serial_shape[1]", cls_name)
        validator.check_int(d_lj_a[0], q, Rel.EQ, "d_LJ_A_shape[0]", cls_name)
        validator.check_int(d_lj_b[0], q, Rel.EQ, "d_LJ_B_shape[0]", cls_name)
        validator.check_int(beta[0], 1, Rel.EQ, "beta_shape", cls_name)
        return [n, 3]

    def infer_dtype(self, uint_crd, ljtype, charge, scaler, nl_numbers, nl_serial, d_lj_a, d_lj_b, beta):
        validator.check_tensor_dtype_valid('uint_crd', uint_crd, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('nl_numbers', nl_numbers, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('nl_serial', nl_serial, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('d_LJ_A', d_lj_a, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('d_LJ_B', d_lj_b, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('beta', beta, [mstype.float32], self.name)
        return mstype.float32


class PMEReciprocalForceUpdate(PrimitiveWithInfer):
    """
    Calculate the reciprocal part of long-range Coulumb force using
    PME(Particle Meshed Ewald) method for pressure. Assume the number of atoms is n.

    The detailed calculation formula of PME(Particle Meshed Ewald) method
    can be found in this paper: A Smooth Particle Mesh Ewald Method. DOI:
    10.1063/1.470117.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms, n.
        beta (float32): the PME beta parameter, determined by the
                       non-bond cutoff value and simulation precision tolerance.
        fftx (int32): the number of points for Fourier transform in dimension X.
        ffty (int32): the number of points for Fourier transform in dimension Y.
        fftz (int32): the number of points for Fourier transform in dimension Z.
        box_length_0 (float32): the value of boxlength idx 0
        box_length_1 (float32): the value of boxlength idx 1
        box_length_2 (float32): the value of boxlength idx 2
        need_update (int32): if need_update = 1, calculate the pressure, default 0.

    Inputs:
        - **uint_crd** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **charge** (Tensor) - The charge carried by each atom.
          The data type is float32 and the shape is :math:`(n,)`
        - **beta** (Tensor) - The PME beta parameter to be updated in pressure calculation.
          The data type is float32 and the shape is :math:`(1,)`

    Outputs:
        - **force** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, beta, fftx, ffty, fftz,
                 box_length_0, box_length_1, box_length_2, need_update=0):
        """Initialize PMEReciprocalForce"""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('beta', beta, float, self.name)
        validator.check_value_type('fftx', fftx, int, self.name)
        validator.check_value_type('ffty', ffty, int, self.name)
        validator.check_value_type('fftz', fftz, int, self.name)
        validator.check_value_type('box_length_0', box_length_0, float, self.name)
        validator.check_value_type('box_length_1', box_length_1, float, self.name)
        validator.check_value_type('box_length_2', box_length_2, float, self.name)
        validator.check_value_type('need_update', need_update, int, self.name)
        self.atom_numbers = atom_numbers
        self.beta = beta
        self.fftx = fftx
        self.ffty = ffty
        self.fftz = fftz
        self.box_length_0 = box_length_0
        self.box_length_1 = box_length_1
        self.box_length_2 = box_length_2
        self.need_update = need_update

        self.init_prim_io_names(inputs=['uint_crd', 'charge', 'beta'],
                                outputs=['force'])
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('beta', self.beta)
        self.add_prim_attr('fftx', self.fftx)
        self.add_prim_attr('ffty', self.ffty)
        self.add_prim_attr('fftz', self.fftz)
        self.add_prim_attr('box_length_0', self.box_length_0)
        self.add_prim_attr('box_length_1', self.box_length_1)
        self.add_prim_attr('box_length_2', self.box_length_2)
        self.add_prim_attr('need_update', self.need_update)

    def infer_shape(self, uint_crd_shape, charge_shape, beta):
        cls_name = self.name
        n = self.atom_numbers
        validator.check_int(len(uint_crd_shape), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(charge_shape), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(beta), 1, Rel.EQ, "beta_dim", cls_name)
        validator.check_int(uint_crd_shape[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd_shape[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(charge_shape[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(beta[0], 1, Rel.EQ, "beta_shape", cls_name)
        return uint_crd_shape

    def infer_dtype(self, uint_crd_type, charge_type, beta):
        validator.check_tensor_dtype_valid('uint_crd', uint_crd_type, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('charge', charge_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('beta', beta, [mstype.float32], self.name)
        return charge_type


class PMEExcludedForceUpdate(PrimitiveWithInfer):
    """
    Calculate the excluded  part of long-range Coulumb force using
    PME(Particle Meshed Ewald) method for pressure. Assume the number of atoms is
    n, and the length of excluded list is E.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms, n.
        excluded_numbers (int32): the length of excluded list, E.
        beta (float32): the PME beta parameter, determined by the
          non-bond cutoff value and simulation precision tolerance.
        need_update (int32): if need_update = 1, calculate the pressure, default 0.

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
        - **beta** (Tensor) - The PME beta parameter to be updated in pressure calculation.
          The data type is float32 and the shape is :math:`(1,)`

    Outputs:
        - **force** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, excluded_numbers, beta, need_update=0):
        """Initialize PMEExcludedForce"""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('excluded_numbers', excluded_numbers, int, self.name)
        validator.check_value_type('beta', beta, float, self.name)
        validator.check_value_type('need_update', need_update, int, self.name)
        self.atom_numbers = atom_numbers
        self.excluded_numbers = excluded_numbers
        self.beta = beta
        self.need_update = need_update
        self.init_prim_io_names(
            inputs=['uint_crd', 'scaler', 'charge', 'excluded_list_start', 'excluded_list',
                    'excluded_atom_numbers', 'beta'],
            outputs=['force'])
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('excluded_numbers', self.excluded_numbers)
        self.add_prim_attr('beta', self.beta)
        self.add_prim_attr('need_update', self.need_update)

    def infer_shape(self, uint_crd_shape, scaler_shape, charge_shape, excluded_list_start_shape, excluded_list_shape,
                    excluded_atom_numbers_shape, beta):
        cls_name = self.name
        n = self.atom_numbers
        e = self.excluded_numbers
        validator.check_int(len(uint_crd_shape), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(scaler_shape), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(charge_shape), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(excluded_list_start_shape), 1, Rel.EQ, "excluded_list_start_dim", cls_name)
        validator.check_int(len(excluded_atom_numbers_shape), 1, Rel.EQ, "excluded_atom_numbers_dim", cls_name)
        validator.check_int(len(excluded_list_shape), 1, Rel.EQ, "excluded_list_dim", cls_name)
        validator.check_int(len(beta), 1, Rel.EQ, "beta_dim", cls_name)
        validator.check_int(uint_crd_shape[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd_shape[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(scaler_shape[0], 3, Rel.EQ, "scaler_shape", cls_name)
        validator.check_int(charge_shape[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(excluded_list_start_shape[0], n, Rel.EQ, "excluded_list_start_shape", cls_name)
        validator.check_int(excluded_atom_numbers_shape[0], n, Rel.EQ, "excluded_atom_numbers_shape", cls_name)
        validator.check_int(excluded_list_shape[0], e, Rel.EQ, "excluded_list_shape", cls_name)
        validator.check_int(beta[0], 1, Rel.EQ, "beta_shape", cls_name)
        return [n, 3]

    def infer_dtype(self, uint_crd_type, scaler_type, charge_type, excluded_list_start_type, excluded_list_type,
                    excluded_atom_numbers_type, beta):
        validator.check_tensor_dtype_valid('scaler', scaler_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('uint_crd', uint_crd_type, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('charge', charge_type, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('excluded_list_start', excluded_list_start_type, [mstype.int32],
                                           self.name)
        validator.check_tensor_dtype_valid('excluded_list', excluded_list_type, [mstype.int32],
                                           self.name)
        validator.check_tensor_dtype_valid('excluded_atom_numbers', excluded_atom_numbers_type, [mstype.int32],
                                           self.name)
        validator.check_tensor_dtype_valid('beta', beta, [mstype.float32], self.name)
        return mstype.float32


class LJForceWithVirialEnergyUpdate(PrimitiveWithInfer):
    """
    Calculate the Lennard-Jones force and PME direct force together for pressure.

    The calculation formula of Lennard-Jones part is the same as operator
    LJForce(), and the PME direct part is within PME method.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms, n.
        cutoff (float32): the square value of cutoff.
        pme_beta (float32): PME beta parameter, same as operator PMEReciprocalForce().
        max_neighbor_numbers (int32): the max neighbor numbers, default 800.
        need_update (int32): if need_update = 1, calculate the pressure, default 0.

    Inputs:
        - **uint_crd** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **LJtype** (Tensor) - The Lennard-Jones type of each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **charge** (Tensor) - The charge carried by each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **scaler** (Tensor) - The scale factor.
          The data type is float32 and the shape is :math:`(3,)`.
        - **nl_numbers** (Tensor) - The each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **nl_serial** (Tensor) - The neighbor list of each atom, the max number is 800.
          The data type is int32 and the shape is :math:`(n, 800)`.
        - **d_LJ_A** (Tensor) - The Lennard-Jones A coefficient of each kind of atom pair.
          The number of atom pair is q. The data type is float32 and the shape is :math:`(q,)`.
        - **d_LJ_B** (Tensor) - The Lennard-Jones B coefficient of each kind of atom pair.
          The number of atom pair is q. The data type is float32 and the shape is :math:`(q,)`.
        - **beta** (Tensor) - The PME beta parameter to be updated in pressure calculation.
          The data type is float32 and the shape is :math:`(1,)`

    Outputs:
        - **frc** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **virial** (Tensor) - The accumulated potential virial for each atom.
          The data type is float32 and the shape is :math:`(n, )`.
        - **atom_energy** (Tensor) - The accumulated potential energy for each atom.
          The data type is float32 and the shape is :math:`(n, )`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, cutoff, pme_beta, max_neighbor_numbers=800, need_update=0):
        """Initialize LJForceWithPMEDirectForce"""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('cutoff', cutoff, float, self.name)
        validator.check_value_type('pme_beta', pme_beta, float, self.name)
        validator.check_value_type('max_neighbor_numbers', max_neighbor_numbers, int, self.name)
        validator.check_value_type('need_update', need_update, int, self.name)
        self.atom_numbers = atom_numbers
        self.cutoff = cutoff
        self.pme_beta = pme_beta
        self.max_neighbor_numbers = max_neighbor_numbers
        self.need_update = need_update
        self.init_prim_io_names(
            inputs=['uint_crd', 'LJtype', 'charge', 'scaler', 'nl_numbers', 'nl_serial', 'd_LJ_A', 'd_LJ_B', 'beta'],
            outputs=['frc', 'virial', 'atom_energy'])
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('cutoff', self.cutoff)
        self.add_prim_attr('pme_beta', self.pme_beta)
        self.add_prim_attr('max_neighbor_numbers', self.max_neighbor_numbers)
        self.add_prim_attr('need_update', self.need_update)

    def infer_shape(self, uint_crd, ljtype, charge, scaler, nl_numbers, nl_serial, d_lj_a, d_lj_b, beta):
        cls_name = self.name
        n = self.atom_numbers
        q = d_lj_a[0]
        m = self.max_neighbor_numbers
        validator.check_int(len(uint_crd), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(ljtype), 1, Rel.EQ, "LJtype_dim", cls_name)
        validator.check_int(len(charge), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(scaler), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(nl_numbers), 1, Rel.EQ, "nl_numbers_dim", cls_name)
        validator.check_int(len(nl_serial), 2, Rel.EQ, "nl_serial_dim", cls_name)
        validator.check_int(len(d_lj_a), 1, Rel.EQ, "d_LJ_A_dim", cls_name)
        validator.check_int(len(d_lj_b), 1, Rel.EQ, "d_LJ_B_dim", cls_name)
        validator.check_int(len(beta), 1, Rel.EQ, "beta_dim", cls_name)
        validator.check_int(uint_crd[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(ljtype[0], n, Rel.EQ, "LJtype_shape", cls_name)
        validator.check_int(charge[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(scaler[0], 3, Rel.EQ, "scaler_shape", cls_name)
        validator.check_int(nl_numbers[0], n, Rel.EQ, "nl_numbers_shape", cls_name)
        validator.check_int(nl_serial[0], n, Rel.EQ, "nl_serial_shape[0]", cls_name)
        validator.check_int(nl_serial[1], m, Rel.EQ, "nl_serial_shape[1]", cls_name)
        validator.check_int(d_lj_a[0], q, Rel.EQ, "d_LJ_A_shape[0]", cls_name)
        validator.check_int(d_lj_b[0], q, Rel.EQ, "d_LJ_B_shape[0]", cls_name)
        validator.check_int(beta[0], 1, Rel.EQ, "beta_shape[0]", cls_name)
        return [n, 3], [n,], [n,]

    def infer_dtype(self, uint_crd, ljtype, charge, scaler, nl_numbers, nl_serial, d_lj_a, d_lj_b, beta):
        validator.check_tensor_dtype_valid('uint_crd', uint_crd, [mstype.uint32], self.name)
        validator.check_tensor_dtype_valid('LJtype', ljtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('charge', charge, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('nl_numbers', nl_numbers, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('nl_serial', nl_serial, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('d_LJ_A', d_lj_a, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('d_LJ_B', d_lj_b, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('beta', beta, [mstype.float32], self.name)
        return mstype.float32, mstype.float32, mstype.float32


class Dihedral14ForceWithAtomEnergyVirial(PrimitiveWithInfer):
    """
    Calculate the Lennard-Jones and Coulumb energy correction and force correction
    for each necessary dihedral 1,4 terms together and add them to the total force
    and potential energy for each atom.

    The calculation formula of force correction is the same as operator
    :class:`Dihedral14LJForceWithDirectCF`, and the energy correction part is the same
    as operator :class:`Dihedral14LJEnergy` and :class:`Dihedral14CFEnergy`.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        nb14_numbers (int32): the number of necessary dihedral 1,4 terms m.
        atom_numbers (int32): the number of atoms n.

    Inputs:
        - **uint_crd_f** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **LJtype** (Tensor) - The Lennard-Jones type of each atom.
          The data type is int32 and the shape is :math:`(n,)`.
        - **charge** (Tensor) - The charge carried by each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **boxlength** (Tensor) - The length of molecular simulation box in 3 dimensions.
          The data type is float32 and the shape is :math:`(3,)`.
        - **a_14** (Tensor) - The first atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **b_14** (Tensor) - The second atom index of each dihedral 1,4 term.
          The data type is int32 and the shape is :math:`(m,)`.
        - **lj_scale_factor** (Tensor) - The scale factor for the
          Lennard-Jones part of force correction of each dihedral 1,4 term.
        - **cf_scale_factor** (Tensor) - The scale factor for the Coulomb force.
          The data type is float32 and the shape is :math:`(m,)`.
        - **LJ_type_A** (Tensor) - The A parameter in Lennard-Jones scheme of each atom pair type.
          The number of atom pair is q. The data type is float32 and the shape is :math:`(q,)`.
        - **LJ_type_B** (Tensor) - The B parameter in Lennard-Jones shceme of each atom pair type.
          The number of atom pair is q. The data type is float32 and the shape is :math:`(q,)`.

    Outputs:
        - **frc** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **atom_energy** (Tensor) - The accumulated potential energy for each atom.
          The data type is float32 and the shape is :math:`(n, )`.
        - **atom_virial** (Tensor) - The accumulated potential virial for each atom.
          The data type is float32 and the shape is :math:`(n, )`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, nb14_numbers, atom_numbers):
        """Initialize Dihedral14LJCFForceWithAtomEnergy"""
        validator.check_value_type('nb14_numbers', nb14_numbers, int, self.name)
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        self.dihedral_14_numbers = nb14_numbers
        self.atom_numbers = atom_numbers

        self.init_prim_io_names(
            inputs=['uint_crd_f', 'LJtype', 'charge', 'boxlength', 'a_14', 'b_14', 'lj_scale_factor',
                    'cf_scale_factor', 'LJ_type_A', 'LJ_type_B'],
            outputs=['frc', 'atom_energy', 'atom_virial'])
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
        validator.check_int(len(lj_type_a_shape), 1, Rel.EQ, "LJ_type_A_dim", cls_name)
        validator.check_int(len(lj_type_b_shape), 1, Rel.EQ, "LJ_type_B_dim", cls_name)

        validator.check_int(uint_crd_f_shape[0], n, Rel.EQ, "uint_crd_f_shape[0]", cls_name)
        validator.check_int(uint_crd_f_shape[1], 3, Rel.EQ, "uint_crd_f_shape[1]", cls_name)
        validator.check_int(ljtype_shape[0], n, Rel.EQ, "LJtype_shape", cls_name)
        validator.check_int(charge_shape[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(boxlength_f_shape[0], 3, Rel.EQ, "boxlength_f_shape", cls_name)
        validator.check_int(lj_type_a_shape[0], q, Rel.EQ, "LJ_type_A_shape", cls_name)
        validator.check_int(lj_type_b_shape[0], q, Rel.EQ, "LJ_type_B_shape", cls_name)
        validator.check_int(a_14_shape[0], m, Rel.EQ, "a_14_shape", cls_name)
        validator.check_int(b_14_shape[0], m, Rel.EQ, "b_14_shape", cls_name)
        validator.check_int(lj_scale_factor_shape[0], m, Rel.EQ, "lj_scale_factor_shape", cls_name)
        validator.check_int(cf_scale_factor_shape[0], m, Rel.EQ, "cf_scale_factor_shape", cls_name)
        return [n, 3], [n,], [n,]

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

        return mstype.float32, mstype.float32, mstype.float32


class PMEEnergyUpdate(PrimitiveWithInfer):
    """
    Calculate the Coulumb energy of the system using PME method for pressure.

    Because there is a large amount of inputs and each of them are related,
    there is no way to construct `Examples` using random methods. For details, refer the webpage `SPONGE in MindSpore
    <https://gitee.com/mindspore/mindscience/blob/master/MindSPONGE/docs/simple_formula.md>`_.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms, n.
        excluded_numbers (int32): the length of excluded list, E.
        beta (float32): the PME beta parameter, determined by the
                       non-bond cutoff value and simulation precision tolerance.
        fftx (int32): the number of points for Fourier transform in dimension X.
        ffty (int32): the number of points for Fourier transform in dimension Y.
        fftz (int32): the number of points for Fourier transform in dimension Z.
        box_length_0 (float32): the value of boxlength idx 0.
        box_length_1 (float32): the value of boxlength idx 1.
        box_length_2 (float32): the value of boxlength idx 2.
        max_neighbor_numbers (int32): the max neighbor numbers, m, default 800.
        need_update (int32): if need_update = 1, calculate the pressure, default 0.

    Inputs:
        - **uint_crd** (Tensor) - The unsigned int coordinates value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`
        - **charge** (Tensor) - The charge carried by each atom.
          The data type is float32 and the shape is :math:`(n,)`
        - **nl_numbers** - (Tensor) - The each atom.
          The data type is int32 and the shape is :math:`(n, 3)`
        - **nl_serial** - (Tensor) - The neighbor list of each atom, the max number is 800.
          The data type is int32 and the shape is :math:`(n, m)`
        - **scaler** (Tensor) - The scale factor between real space
          coordinates and its unsigned int value. The data type is float32 and the shape is :math:`(3,)`
        - **excluded_list_start** (Tensor) - The start excluded index
          in excluded list for each atom. The data type is int32 and the shape is :math:`(n,)`
        - **excluded_list** (Tensor) - The contiguous join of excluded
          list of each atom. E is the number of excluded atoms. The data type is int32 and the shape is :math:`(E,)`
        - **excluded_atom_numbers** (Tensor) - The number of atom excluded
          in excluded list for each atom. The data type is int32 and the shape is :math:`(n,)`
        - **factor** (Tensor) - The factor parameter to be updated in pressure calculation.
          The data type is float32 and the shape is :math:`(1,)`
        - **beta** (Tensor) - The PME beta parameter to be updated in pressure calculation.
          The data type is float32 and the shape is :math:`(1,)`

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
                 box_length_2, max_neighbor_numbers=800, need_update=0):
        """Initialize PMEEnergyUpdate"""
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('excluded_numbers', excluded_numbers, int, self.name)
        validator.check_value_type('beta', beta, float, self.name)
        validator.check_value_type('fftx', fftx, int, self.name)
        validator.check_value_type('ffty', ffty, int, self.name)
        validator.check_value_type('fftz', fftz, int, self.name)
        validator.check_value_type('box_length_0', box_length_0, float, self.name)
        validator.check_value_type('box_length_1', box_length_1, float, self.name)
        validator.check_value_type('box_length_2', box_length_2, float, self.name)
        validator.check_value_type('max_neighbor_numbers', max_neighbor_numbers, int, self.name)
        validator.check_value_type('need_update', need_update, int, self.name)
        self.atom_numbers = atom_numbers
        self.excluded_numbers = excluded_numbers
        self.beta = beta
        self.fftx = fftx
        self.ffty = ffty
        self.fftz = fftz
        self.box_length_0 = box_length_0
        self.box_length_1 = box_length_1
        self.box_length_2 = box_length_2
        self.max_neighbor_numbers = max_neighbor_numbers
        self.need_update = need_update
        self.init_prim_io_names(
            inputs=['uint_crd', 'charge', 'nl_numbers', 'nl_serial', 'scaler', 'excluded_list_start',
                    'excluded_list', 'excluded_atom_numbers', 'factor', 'beta'],
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
        self.add_prim_attr('max_neighbor_numbers', self.max_neighbor_numbers)
        self.add_prim_attr('need_update', self.need_update)

    def infer_shape(self, uint_crd, charge, nl_numbers, nl_serial, scaler, excluded_list_start,
                    excluded_list, excluded_atom_numbers, factor, beta):
        cls_name = self.name
        n = self.atom_numbers
        m = self.max_neighbor_numbers
        e = self.excluded_numbers
        validator.check_int(len(uint_crd), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(charge), 1, Rel.EQ, "charge_dim", cls_name)
        validator.check_int(len(nl_numbers), 1, Rel.EQ, "nl_numbers_dim", cls_name)
        validator.check_int(len(nl_serial), 2, Rel.EQ, "nl_serial_dim", cls_name)
        validator.check_int(len(excluded_list_start), 1, Rel.EQ, "excluded_list_start_dim", cls_name)
        validator.check_int(len(excluded_atom_numbers), 1, Rel.EQ, "excluded_atom_numbers_dim", cls_name)
        validator.check_int(len(excluded_list), 1, Rel.EQ, "excluded_list_dim", cls_name)
        validator.check_int(len(scaler), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(factor), 1, Rel.EQ, "factor_dim", cls_name)
        validator.check_int(len(beta), 1, Rel.EQ, "beta_dim", cls_name)
        validator.check_int(uint_crd[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(charge[0], n, Rel.EQ, "charge_shape", cls_name)
        validator.check_int(nl_numbers[0], n, Rel.EQ, "nl_numbers_shape[0]", cls_name)
        validator.check_int(nl_serial[0], n, Rel.EQ, "nl_serial_shape[0]", cls_name)
        validator.check_int(nl_serial[1], m, Rel.EQ, "nl_serial_shape[1]", cls_name)
        validator.check_int(excluded_list_start[0], n, Rel.EQ, "excluded_list_start_shape", cls_name)
        validator.check_int(excluded_atom_numbers[0], n, Rel.EQ, "excluded_atom_numbers_shape", cls_name)
        validator.check_int(excluded_list[0], e, Rel.EQ, "excluded_list_shape", cls_name)
        validator.check_int(factor[0], 1, Rel.EQ, "factor_shape", cls_name)
        validator.check_int(beta[0], 1, Rel.EQ, "beta_shape", cls_name)
        validator.check_int(scaler[0], 3, Rel.EQ, "scaler_shape", cls_name)
        return [1,], [1,], [1,], [1,]

    def infer_dtype(self, uint_crd, charge, nl_numbers, nl_serial, scaler, excluded_list_start,
                    excluded_list, excluded_atom_numbers, factor, beta):
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
        validator.check_tensor_dtype_valid('factor', factor, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('beta', beta, [mstype.float32], self.name)
        return charge, charge, charge, charge


class ConstrainForceCycle(PrimitiveWithInfer):
    """
    Calculate the constraint force in each iteration.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        constrain_pair_numbers (int32): the number of constrain pairs m.

    Inputs:
        - **uint_crd** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler** (Tensor) - The 3-D scale factor (x, y, z),
          The data type is float32 and the shape is :math:`(3,)`.
        - **pair_dr** (Tensor) - The displacement vector of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m, 3)`.
        - **atom_i_serials** (Tensor) - The first atom index of each constrained atom pair.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_j_serials** (Tensor) - The second atom index of each constrained atom pair.
          The data type is int32 and the shape is :math:`(m,)`.
        - **constant_rs** (Tensor) - The constrained distance of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m,)`.
        - **constrain_ks** (Tensor) - The coefficient of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **test_frc** (Tensor) - The constraint force.
          The data type is float32 and the shape is :math:`(n, 3)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, constrain_pair_numbers):
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('constrain_pair_numbers', constrain_pair_numbers, int, self.name)
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
        m = self.constrain_pair_numbers
        validator.check_int(len(uint_crd_shape), 2, Rel.EQ, "uint_crd_dim", cls_name)
        validator.check_int(len(scaler_shape), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(pair_dr_shape), 2, Rel.EQ, "pair_dr_dim", cls_name)
        validator.check_int(len(atom_i_serials_shape), 1, Rel.EQ, "atom_i_serials_dim", cls_name)
        validator.check_int(len(atom_j_serials_shape), 1, Rel.EQ, "atom_j_serials_dim", cls_name)
        validator.check_int(len(constant_rs_shape), 1, Rel.EQ, "constant_rs_dim", cls_name)
        validator.check_int(len(constrain_ks_shape), 1, Rel.EQ, "constrain_ks_dim", cls_name)

        validator.check_int(uint_crd_shape[0], n, Rel.EQ, "uint_crd_shape[0]", cls_name)
        validator.check_int(uint_crd_shape[1], 3, Rel.EQ, "uint_crd_shape[1]", cls_name)
        validator.check_int(scaler_shape[0], 3, Rel.EQ, "scaler_shape", cls_name)
        validator.check_int(pair_dr_shape[0], m, Rel.EQ, "pair_dr_shape[0]", cls_name)
        validator.check_int(pair_dr_shape[1], 3, Rel.EQ, "pair_dr_shape[1]", cls_name)
        validator.check_int(atom_i_serials_shape[0], m, Rel.EQ, "atom_i_serials_shape[0]", cls_name)
        validator.check_int(atom_j_serials_shape[0], m, Rel.EQ, "atom_j_serials_shape[0]", cls_name)
        validator.check_int(constant_rs_shape[0], m, Rel.EQ, "constant_rs_shape[0]", cls_name)
        validator.check_int(constrain_ks_shape[0], m, Rel.EQ, "constrain_ks_shape[0]", cls_name)
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


class ConstrainForceVirial(PrimitiveWithInfer):
    """
    Calculate the constraint force and virial in a step with iteration numbers.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        constrain_pair_numbers (int32): the number of constrain pairs m.
        iteration_numbers (int32): the number of iteration numbers p.
        half_exp_gamma_plus_half (float32): half exp_gamma plus half q.

    Inputs:
        - **crd** (Tensor) - The coordinate of each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **quarter_cof** (Tensor) - The 3-D scale factor.
          The data type is float32 and the shape is :math:`(3,)`.
        - **mass_inverse** (Tensor) - The inverse value of mass of each atom.
          The data type is float32 and the shape is :math:`(n,)`.
        - **scaler** (Tensor) - The 3-D scale factor (x, y, z),
          The data type is float32 and the shape is :math:`(3,)`.
        - **pair_dr** (Tensor) - The displacement vector of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m, 3)`.
        - **atom_i_serials** (Tensor) - The first atom index of each constrained atom pair.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_j_serials** (Tensor) - The second atom index of each constrained atom pair.
          The data type is int32 and the shape is :math:`(m,)`.
        - **constant_rs** (Tensor) - The constrained distance of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m,)`.
        - **constrain_ks** (Tensor) - The coefficient of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **uint_crd** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **frc** (Tensor) - The force felt by each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **virial** (Tensor) - The constraint virial on each atom.
          The data type is float32 and the shape is :math:`(m,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, constrain_pair_numbers, iteration_numbers, half_exp_gamma_plus_half):
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('constrain_pair_numbers', constrain_pair_numbers, int, self.name)
        validator.check_value_type('iteration_numbers', iteration_numbers, int, self.name)
        validator.check_value_type('half_exp_gamma_plus_half', half_exp_gamma_plus_half, float, self.name)
        self.atom_numbers = atom_numbers
        self.constrain_pair_numbers = constrain_pair_numbers
        self.iteration_numbers = iteration_numbers
        self.half_exp_gamma_plus_half = half_exp_gamma_plus_half
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('constrain_pair_numbers', self.constrain_pair_numbers)
        self.add_prim_attr('iteration_numbers', self.iteration_numbers)
        self.add_prim_attr('half_exp_gamma_plus_half', self.half_exp_gamma_plus_half)

        self.init_prim_io_names(
            inputs=['crd', 'quarter_cof', 'mass_inverse',
                    'scaler', 'pair_dr', 'atom_i_serials', 'atom_j_serials',
                    'constant_rs', 'constrain_ks', 'test_frc', 'atom_virial'],
            outputs=['uint_crd', 'frc', 'virial'])

    def infer_shape(self, crd, quarter_cof, mass_inverse, scaler_shape, pair_dr_shape, atom_i_serials_shape,
                    atom_j_serials_shape, constant_rs_shape, constrain_ks_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.constrain_pair_numbers
        validator.check_int(len(crd), 2, Rel.EQ, "crd_dim", cls_name)
        validator.check_int(len(quarter_cof), 1, Rel.EQ, "quarter_cof_dim", cls_name)
        validator.check_int(len(mass_inverse), 1, Rel.EQ, "mass_inverse_dim", cls_name)
        validator.check_int(len(scaler_shape), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(pair_dr_shape), 2, Rel.EQ, "pair_dr_dim", cls_name)
        validator.check_int(len(atom_i_serials_shape), 1, Rel.EQ, "atom_i_serials_dim", cls_name)
        validator.check_int(len(atom_j_serials_shape), 1, Rel.EQ, "atom_j_serials_dim", cls_name)
        validator.check_int(len(constant_rs_shape), 1, Rel.EQ, "constant_rs_dim", cls_name)
        validator.check_int(len(constrain_ks_shape), 1, Rel.EQ, "constrain_ks_dim", cls_name)
        validator.check_int(crd[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(quarter_cof[0], 3, Rel.EQ, "quarter_cof_shape", cls_name)
        validator.check_int(mass_inverse[0], n, Rel.EQ, "quarter_cof_shape", cls_name)
        validator.check_int(scaler_shape[0], 3, Rel.EQ, "scaler_shape", cls_name)
        validator.check_int(pair_dr_shape[0], m, Rel.EQ, "pair_dr_shape[0]", cls_name)
        validator.check_int(pair_dr_shape[1], 3, Rel.EQ, "pair_dr_shape[1]", cls_name)
        validator.check_int(atom_i_serials_shape[0], m, Rel.EQ, "atom_i_serials_shape[0]", cls_name)
        validator.check_int(atom_j_serials_shape[0], m, Rel.EQ, "atom_j_serials_shape[0]", cls_name)
        validator.check_int(constant_rs_shape[0], m, Rel.EQ, "constant_rs_shape[0]", cls_name)
        validator.check_int(constrain_ks_shape[0], m, Rel.EQ, "constrain_ks_shape[0]", cls_name)
        return [n, 3], [n, 3], [m,]

    def infer_dtype(self, crd, quarter_cof, mass_inverse, scaler_dtype, pair_dr_dtype, atom_i_serials_dtype,
                    atom_j_serials_dtype, constant_rs_dtype, constrain_ks_dtype):
        validator.check_tensor_dtype_valid('crd', crd, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('quarter_cof', quarter_cof, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('mass_inverse', mass_inverse, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pair_dr', pair_dr_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_i_serials', atom_i_serials_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_j_serials', atom_j_serials_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('constant_rs', constant_rs_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('constrain_ks', constrain_ks_dtype, [mstype.float32], self.name)
        return mstype.uint32, mstype.float32, mstype.float32


class ConstrainForce(PrimitiveWithInfer):
    """
    Calculate the constraint force in a step with iteration numbers.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        constrain_pair_numbers (int32): the number of constrain pairs m.
        iteration_numbers (int32): the number of iteration numbers p.
        half_exp_gamma_plus_half (float32): half exp_gamma plus half q.

    Inputs:
        - **uint_crd** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler** (Tensor) - The 3-D scale factor (x, y, z),
          The data type is float32 and the shape is :math:`(3,)`.
        - **pair_dr** (Tensor) - The displacement vector of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m, 3)`.
        - **atom_i_serials** (Tensor) - The first atom index of each constrained atom pair.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_j_serials** (Tensor) - The second atom index of each constrained atom pair.
          The data type is int32 and the shape is :math:`(m,)`.
        - **constant_rs** (Tensor) - The constrained distance of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m,)`.
        - **constrain_ks** (Tensor) - The coefficient of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m,)`.

    Outputs:
        - **uint_crd** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **frc** (Tensor) - The constraint force on each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **virial** (Tensor) - The constraint virial on each atom and it is zero.
          The data type is float32 and the shape is :math:`(m,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, constrain_pair_numbers, iteration_numbers, half_exp_gamma_plus_half):
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('constrain_pair_numbers', constrain_pair_numbers, int, self.name)
        validator.check_value_type('iteration_numbers', iteration_numbers, int, self.name)
        validator.check_value_type('half_exp_gamma_plus_half', half_exp_gamma_plus_half, float, self.name)
        self.atom_numbers = atom_numbers
        self.constrain_pair_numbers = constrain_pair_numbers
        self.iteration_numbers = iteration_numbers
        self.half_exp_gamma_plus_half = half_exp_gamma_plus_half
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('constrain_pair_numbers', self.constrain_pair_numbers)
        self.add_prim_attr('iteration_numbers', self.iteration_numbers)
        self.add_prim_attr('half_exp_gamma_plus_half', self.half_exp_gamma_plus_half)

        self.init_prim_io_names(
            inputs=['crd', 'quarter_cof', 'mass_inverse',
                    'scaler', 'pair_dr', 'atom_i_serials', 'atom_j_serials', 'constant_rs', 'constrain_ks'],
            outputs=['uint_crd', 'frc', 'virial'])

    def infer_shape(self, crd, quarter_cof, mass_inverse, scaler_shape, pair_dr_shape, atom_i_serials_shape,
                    atom_j_serials_shape, constant_rs_shape, constrain_ks_shape):
        cls_name = self.name
        n = self.atom_numbers
        m = self.constrain_pair_numbers
        validator.check_int(len(crd), 2, Rel.EQ, "crd_dim", cls_name)
        validator.check_int(len(quarter_cof), 1, Rel.EQ, "quarter_cof_dim", cls_name)
        validator.check_int(len(mass_inverse), 1, Rel.EQ, "mass_inverse_dim", cls_name)
        validator.check_int(len(scaler_shape), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(pair_dr_shape), 2, Rel.EQ, "pair_dr_dim", cls_name)
        validator.check_int(len(atom_i_serials_shape), 1, Rel.EQ, "atom_i_serials_dim", cls_name)
        validator.check_int(len(atom_j_serials_shape), 1, Rel.EQ, "atom_j_serials_dim", cls_name)
        validator.check_int(len(constant_rs_shape), 1, Rel.EQ, "constant_rs_dim", cls_name)
        validator.check_int(len(constrain_ks_shape), 1, Rel.EQ, "constrain_ks_dim", cls_name)
        validator.check_int(crd[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(quarter_cof[0], 3, Rel.EQ, "quarter_cof_shape", cls_name)
        validator.check_int(mass_inverse[0], n, Rel.EQ, "quarter_cof_shape", cls_name)
        validator.check_int(scaler_shape[0], 3, Rel.EQ, "scaler_shape", cls_name)
        validator.check_int(pair_dr_shape[0], m, Rel.EQ, "pair_dr_shape[0]", cls_name)
        validator.check_int(pair_dr_shape[1], 3, Rel.EQ, "pair_dr_shape[1]", cls_name)
        validator.check_int(atom_i_serials_shape[0], m, Rel.EQ, "atom_i_serials_shape[0]", cls_name)
        validator.check_int(atom_j_serials_shape[0], m, Rel.EQ, "atom_j_serials_shape[0]", cls_name)
        validator.check_int(constant_rs_shape[0], m, Rel.EQ, "constant_rs_shape[0]", cls_name)
        validator.check_int(constrain_ks_shape[0], m, Rel.EQ, "constrain_ks_shape[0]", cls_name)
        return [n, 3], [n, 3], [m,]

    def infer_dtype(self, crd, quarter_cof, mass_inverse, scaler_dtype, pair_dr_dtype, atom_i_serials_dtype,
                    atom_j_serials_dtype, constant_rs_dtype, constrain_ks_dtype):
        validator.check_tensor_dtype_valid('crd', crd, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('quarter_cof', quarter_cof, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('mass_inverse', mass_inverse, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pair_dr', pair_dr_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_i_serials', atom_i_serials_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_j_serials', atom_j_serials_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('constant_rs', constant_rs_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('constrain_ks', constrain_ks_dtype, [mstype.float32], self.name)
        return mstype.uint32, mstype.float32, mstype.float32


class Constrain(PrimitiveWithInfer):
    """
    Calculate the constraint force and virial depends on pressure calculation.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        atom_numbers (int32): the number of atoms n.
        constrain_pair_numbers (int32): the number of constrain pairs m.
        iteration_numbers (int32): the number of iteration numbers p.
        half_exp_gamma_plus_half (float32): half exp_gamma plus half q.

    Inputs:
        - **uint_crd** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **scaler** (Tensor) - The 3-D scale factor (x, y, z),
          The data type is float32 and the shape is :math:`(3,)`.
        - **pair_dr** (Tensor) - The displacement vector of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m, 3)`.
        - **atom_i_serials** (Tensor) - The first atom index of each constrained atom pair.
          The data type is int32 and the shape is :math:`(m,)`.
        - **atom_j_serials** (Tensor) - The second atom index of each constrained atom pair.
          The data type is int32 and the shape is :math:`(m,)`.
        - **constant_rs** (Tensor) - The constrained distance of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m,)`.
        - **constrain_ks** (Tensor) - The coefficient of each constrained atom pair.
          The data type is float32 and the shape is :math:`(m,)`.
        - **need_pressure** (Tensor) - If need pressure, 1 else 0.
          The data type is int32 and the shape is :math:`(1,)`.

    Outputs:
        - **uint_crd** (Tensor) - The unsigned int coordinate value of each atom.
          The data type is uint32 and the shape is :math:`(n, 3)`.
        - **frc** (Tensor) - The constraint force on each atom.
          The data type is float32 and the shape is :math:`(n, 3)`.
        - **virial** (Tensor) - The constraint virial on each atom.
          The data type is float32 and the shape is :math:`(m,)`.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, atom_numbers, constrain_pair_numbers, iteration_numbers, half_exp_gamma_plus_half,
                 update_interval=10):
        validator.check_value_type('atom_numbers', atom_numbers, int, self.name)
        validator.check_value_type('constrain_pair_numbers', constrain_pair_numbers, int, self.name)
        validator.check_value_type('iteration_numbers', iteration_numbers, int, self.name)
        validator.check_value_type('half_exp_gamma_plus_half', half_exp_gamma_plus_half, float, self.name)
        validator.check_value_type('update_interval', update_interval, int, self.name)
        self.atom_numbers = atom_numbers
        self.constrain_pair_numbers = constrain_pair_numbers
        self.iteration_numbers = iteration_numbers
        self.half_exp_gamma_plus_half = half_exp_gamma_plus_half
        self.update_interval = update_interval
        self.add_prim_attr('atom_numbers', self.atom_numbers)
        self.add_prim_attr('constrain_pair_numbers', self.constrain_pair_numbers)
        self.add_prim_attr('iteration_numbers', self.iteration_numbers)
        self.add_prim_attr('half_exp_gamma_plus_half', self.half_exp_gamma_plus_half)
        self.add_prim_attr('update_interval', self.update_interval)

        self.init_prim_io_names(
            inputs=['crd', 'quarter_cof', 'mass_inverse',
                    'scaler', 'pair_dr', 'atom_i_serials', 'atom_j_serials',
                    'constant_rs', 'constrain_ks', 'need_pressure'],
            outputs=['uint_crd', 'frc', 'virial'])

    def infer_shape(self, crd, quarter_cof, mass_inverse, scaler_shape, pair_dr_shape, atom_i_serials_shape,
                    atom_j_serials_shape, constant_rs_shape, constrain_ks_shape, need_pressure):
        cls_name = self.name
        n = self.atom_numbers
        m = self.constrain_pair_numbers
        validator.check_int(len(crd), 2, Rel.EQ, "crd_dim", cls_name)
        validator.check_int(len(quarter_cof), 1, Rel.EQ, "quarter_cof_dim", cls_name)
        validator.check_int(len(mass_inverse), 1, Rel.EQ, "mass_inverse_dim", cls_name)
        validator.check_int(len(scaler_shape), 1, Rel.EQ, "scaler_dim", cls_name)
        validator.check_int(len(pair_dr_shape), 2, Rel.EQ, "pair_dr_dim", cls_name)
        validator.check_int(len(atom_i_serials_shape), 1, Rel.EQ, "atom_i_serials_dim", cls_name)
        validator.check_int(len(atom_j_serials_shape), 1, Rel.EQ, "atom_j_serials_dim", cls_name)
        validator.check_int(len(constant_rs_shape), 1, Rel.EQ, "constant_rs_dim", cls_name)
        validator.check_int(len(constrain_ks_shape), 1, Rel.EQ, "constrain_ks_dim", cls_name)
        validator.check_int(len(need_pressure), 1, Rel.LE, "need_pressure_dim", cls_name)
        validator.check_int(crd[0], n, Rel.EQ, "crd_shape[0]", cls_name)
        validator.check_int(crd[1], 3, Rel.EQ, "crd_shape[1]", cls_name)
        validator.check_int(quarter_cof[0], 3, Rel.EQ, "quarter_cof_shape", cls_name)
        validator.check_int(mass_inverse[0], n, Rel.EQ, "quarter_cof_shape", cls_name)
        validator.check_int(scaler_shape[0], 3, Rel.EQ, "scaler_shape", cls_name)
        validator.check_int(pair_dr_shape[0], m, Rel.EQ, "pair_dr_shape[0]", cls_name)
        validator.check_int(pair_dr_shape[1], 3, Rel.EQ, "pair_dr_shape[1]", cls_name)
        validator.check_int(atom_i_serials_shape[0], m, Rel.EQ, "atom_i_serials_shape[0]", cls_name)
        validator.check_int(atom_j_serials_shape[0], m, Rel.EQ, "atom_j_serials_shape[0]", cls_name)
        validator.check_int(constant_rs_shape[0], m, Rel.EQ, "constant_rs_shape[0]", cls_name)
        validator.check_int(constrain_ks_shape[0], m, Rel.EQ, "constrain_ks_shape[0]", cls_name)
        return [n, 3], [n, 3], [m,]

    def infer_dtype(self, crd, quarter_cof, mass_inverse, scaler_dtype, pair_dr_dtype, atom_i_serials_dtype,
                    atom_j_serials_dtype, constant_rs_dtype, constrain_ks_dtype, need_pressure):
        validator.check_tensor_dtype_valid('crd', crd, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('quarter_cof', quarter_cof, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('mass_inverse', mass_inverse, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('scaler', scaler_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pair_dr', pair_dr_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('atom_i_serials', atom_i_serials_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('atom_j_serials', atom_j_serials_dtype, [mstype.int32], self.name)
        validator.check_tensor_dtype_valid('constant_rs', constant_rs_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('constrain_ks', constrain_ks_dtype, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid('need_pressure', need_pressure, [mstype.int32], self.name)
        return mstype.uint32, mstype.float32, mstype.float32

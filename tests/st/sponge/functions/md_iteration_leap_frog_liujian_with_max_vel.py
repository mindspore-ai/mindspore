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
'''md iteration leap frog liujian with max velocity limitation'''
import mindspore.numpy as np
import mindspore.ops as ops

from .common import get_full_tensor

standard_normal = ops.StandardNormal()
standard_normal.add_prim_attr("use_curand", True)


def md_iteration_leap_frog_liujian_with_max_vel(atom_numbers, half_dt, dt, exp_gamma, inverse_mass,
                                                sqrt_mass_inverse, velocity, crd, frc, acc, max_vel):
    """
    One step of classical leap frog algorithm to solve the finite difference
    Hamiltonian equations of motion for certain system, using Langevin dynamics
    with Liu's thermostat scheme. Assume the number of atoms is N and the target
    control temperature is T.

    Detailed iteration formula can be found in this paper: A unified thermostat
    scheme for efficient configurational sampling for classical/quantum canonical
    ensembles via molecular dynamics. DOI: 10.1063/1.4991621.

    Args:
        atom_numbers (int): the number of atoms N.
        half_dt (float): half of time step for finite difference.
        dt (float): time step for finite difference.
        exp_gamma (float): parameter in Liu's dynamic, equals
            exp (-gamma_ln * dt), where gamma_ln is the firction factor in Langvin
            dynamics.
        inverse_mass (Tensor, float32): [N,], the inverse value of
            mass of each atom.
        sqrt_mass_inverse (Tensor, float32): [N,], the inverse square root value
            of effect mass in Liu's dynamics of each atom.
        velocity (Tensor, float32): [N, 3], the velocity of each atom.
        crd (Tensor, float32): [N, 3], the coordinate of each atom.
        frc (Tensor, float32): [N, 3], the force felt by each atom.
        acc (Tensor, float32): [N, 3], the acceleration of each atom.
        max_vel (flaot32): the max velocity of each atom.

    Outputs:
        velocity (float)
        updated_crd (float)
        acc (float)
        frc (float)

    Supported Platforms:
        ``GPU``
    """
    acc = np.tile(np.expand_dims(inverse_mass, 1), (1, 3)) * frc
    velocity = velocity + get_full_tensor((atom_numbers, 3), dt) * acc

    cur_vel = np.sqrt(np.sum(velocity ** 2, -1, keepdims=True))
    velocity = np.where(cur_vel < max_vel, velocity, max_vel / cur_vel * velocity)

    updated_crd = crd + get_full_tensor((atom_numbers, 3), half_dt) * velocity
    velocity = exp_gamma * velocity + np.expand_dims(sqrt_mass_inverse, 1) * standard_normal((atom_numbers, 3))
    updated_crd = updated_crd + half_dt * velocity
    return velocity, updated_crd, acc

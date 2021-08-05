/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/simple_constrain/last_crd_to_dr_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void Last_Crd_To_dr(const int constarin_pair_numbers, const VECTOR *atom_crd,
                               const VECTOR *quarter_crd_to_uint_crd_cof, const VECTOR *uint_dr_to_dr,
                               CONSTRAIN_PAIR *constrain_pair, VECTOR *pair_dr) {
  int pair_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (pair_i < constarin_pair_numbers) {
    INT_VECTOR tempi;
    INT_VECTOR tempj;
    UNSIGNED_INT_VECTOR uint_crd_i;
    UNSIGNED_INT_VECTOR uint_crd_j;
    CONSTRAIN_PAIR cp = constrain_pair[pair_i];
    VECTOR dr;

    tempi.int_x = atom_crd[cp.atom_i_serial].x * quarter_crd_to_uint_crd_cof[0].x;
    tempi.int_y = atom_crd[cp.atom_i_serial].y * quarter_crd_to_uint_crd_cof[0].y;
    tempi.int_z = atom_crd[cp.atom_i_serial].z * quarter_crd_to_uint_crd_cof[0].z;

    tempj.int_x = atom_crd[cp.atom_j_serial].x * quarter_crd_to_uint_crd_cof[0].x;
    tempj.int_y = atom_crd[cp.atom_j_serial].y * quarter_crd_to_uint_crd_cof[0].y;
    tempj.int_z = atom_crd[cp.atom_j_serial].z * quarter_crd_to_uint_crd_cof[0].z;

    uint_crd_i.uint_x = tempi.int_x << 2;
    uint_crd_i.uint_y = tempi.int_y << 2;
    uint_crd_i.uint_z = tempi.int_z << 2;

    uint_crd_j.uint_x = tempj.int_x << 2;
    uint_crd_j.uint_y = tempj.int_y << 2;
    uint_crd_j.uint_z = tempj.int_z << 2;

    dr.x = (static_cast<int>(uint_crd_i.uint_x - uint_crd_j.uint_x)) * uint_dr_to_dr[0].x;
    dr.y = (static_cast<int>(uint_crd_i.uint_y - uint_crd_j.uint_y)) * uint_dr_to_dr[0].y;
    dr.z = (static_cast<int>(uint_crd_i.uint_z - uint_crd_j.uint_z)) * uint_dr_to_dr[0].z;

    pair_dr[pair_i] = dr;
  }
}

void lastcrdtodr(int constrain_pair_numbers, const float *atom_crd_f, const float *quarter_crd_to_uint_crd_cof_f,
                 const float *uint_dr_to_dr_f, float *constrain_pair_f, const int *atom_i_serials,
                 const int *atom_j_serials, const float *constant_rs, const float *constrain_ks, float *pair_dr_f,
                 cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3 * constrain_pair_numbers) / 128), 128, 0, stream>>>(
    3 * constrain_pair_numbers, pair_dr_f, 0.);
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(constrain_pair_numbers) / 128);
  const VECTOR *atom_crd = reinterpret_cast<const VECTOR *>(atom_crd_f);
  const VECTOR *quarter_crd_to_uint_crd_cof = reinterpret_cast<const VECTOR *>(quarter_crd_to_uint_crd_cof_f);
  const VECTOR *uint_dr_to_dr = reinterpret_cast<const VECTOR *>(uint_dr_to_dr_f);

  CONSTRAIN_PAIR *constrain_pair = reinterpret_cast<CONSTRAIN_PAIR *>(constrain_pair_f);

  VECTOR *pair_dr = reinterpret_cast<VECTOR *>(pair_dr_f);

  construct_constrain_pair<<<ceilf(static_cast<float>(constrain_pair_numbers) / 128), 128, 0, stream>>>(
    constrain_pair_numbers, atom_i_serials, atom_j_serials, constant_rs, constrain_ks, constrain_pair);

  Last_Crd_To_dr<<<block_per_grid, thread_per_block, 0, stream>>>(constrain_pair_numbers, atom_crd,
                                                                  quarter_crd_to_uint_crd_cof, uint_dr_to_dr,
                                                                  constrain_pair, pair_dr);
  return;
}

void lastcrdtodr(int constrain_pair_numbers, const float *atom_crd_f, const float *quarter_crd_to_uint_crd_cof_f,
                 const float *uint_dr_to_dr_f, float *constrain_pair_f, const int *atom_i_serials,
                 const int *atom_j_serials, const float *constant_rs, const float *constrain_ks, float *pair_dr_f,
                 cudaStream_t stream);

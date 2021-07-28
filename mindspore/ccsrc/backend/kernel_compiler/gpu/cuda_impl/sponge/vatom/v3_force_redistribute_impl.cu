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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/vatom/v3_force_redistribute_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

__global__ void v3_Force_Redistribute(const int virtual_numbers, const VIRTUAL_TYPE_3 *v_info,
                                      const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR scaler, VECTOR *force) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < virtual_numbers) {
    VIRTUAL_TYPE_3 v_temp = v_info[i];
    int atom_v = v_temp.virtual_atom;
    int atom_1 = v_temp.from_1;
    int atom_2 = v_temp.from_2;
    int atom_3 = v_temp.from_3;
    float d = v_temp.d;
    float k = v_temp.k;
    VECTOR force_v = force[atom_v];

    UNSIGNED_INT_VECTOR uint_r1 = uint_crd[atom_1];
    UNSIGNED_INT_VECTOR uint_r2 = uint_crd[atom_2];
    UNSIGNED_INT_VECTOR uint_r3 = uint_crd[atom_3];
    UNSIGNED_INT_VECTOR uint_rv = uint_crd[atom_v];

    VECTOR r21 = Get_Periodic_Displacement(uint_r2, uint_r1, scaler);
    VECTOR r32 = Get_Periodic_Displacement(uint_r3, uint_r2, scaler);
    VECTOR rv1 = Get_Periodic_Displacement(uint_rv, uint_r1, scaler);

    VECTOR temp = r21 + k * r32;
    float factor = d * rnorm3df(temp.x, temp.y, temp.z);

    temp = (rv1 * force_v) / (rv1 * rv1) * rv1;
    temp = factor * (force_v - temp);
    VECTOR force_1 = force_v - temp;
    VECTOR force_2 = (1 - k) * temp;
    VECTOR force_3 = k * temp;

    atomicAdd(&force[atom_1].x, force_1.x);
    atomicAdd(&force[atom_1].y, force_1.y);
    atomicAdd(&force[atom_1].z, force_1.z);

    atomicAdd(&force[atom_2].x, force_2.x);
    atomicAdd(&force[atom_2].y, force_2.y);
    atomicAdd(&force[atom_2].z, force_2.z);

    atomicAdd(&force[atom_3].x, force_3.x);
    atomicAdd(&force[atom_3].y, force_3.y);
    atomicAdd(&force[atom_3].z, force_3.z);

    force_v.x = 0.0f;
    force_v.y = 0.0f;
    force_v.z = 0.0f;
    force[atom_v] = force_v;
  }
}

void v3ForceRedistribute(int atom_numbers, int virtual_numbers, const float *v_info_f, const int *uint_crd_f,
                         const float *scaler_f, float *frc_f, cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, frc_f, 0.);
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(virtual_numbers) / 128);
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));

  VECTOR *scaler = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(scaler_f));
  VECTOR *frc = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(frc_f));
  VIRTUAL_TYPE_3 *v_info = const_cast<VIRTUAL_TYPE_3 *>(reinterpret_cast<const VIRTUAL_TYPE_3 *>(v_info_f));

  v3_Force_Redistribute<<<block_per_grid, thread_per_block, 0, stream>>>(virtual_numbers, v_info, uint_crd, scaler[0],
                                                                         frc);

  return;
}

void v3ForceRedistribute(int atom_numbers, int virtual_numbers, const float *v_info_f, const int *uint_crd_f,
                         const float *scaler_f, float *frc_f, cudaStream_t stream);

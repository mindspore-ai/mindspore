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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/restrain/restrain_force_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

__global__ void restrainforcekernel(int restrain_numbers, const int *restrain_list, const UNSIGNED_INT_VECTOR *uint_crd,
                                    const UNSIGNED_INT_VECTOR *uint_crd_ref, const float factor, const VECTOR *scaler,
                                    VECTOR *frc) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < restrain_numbers) {
    int atom_i = restrain_list[i];
    VECTOR dr = Get_Periodic_Displacement(uint_crd_ref[atom_i], uint_crd[atom_i], scaler[0]);

    atomicAdd(&frc[atom_i].x, factor * dr.x);
    atomicAdd(&frc[atom_i].y, factor * dr.y);
    atomicAdd(&frc[atom_i].z, factor * dr.z);
  }
}

void restrainforce(int restrain_numbers, int atom_numbers, const int *restrain_list, const int *uint_crd_f,
                   const int *uint_crd_ref_f, const float factor, const float *scaler_f, float *frc_f,
                   cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, frc_f, 0.);
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(restrain_numbers) / 128);
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));

  UNSIGNED_INT_VECTOR *uint_crd_ref =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_ref_f));

  VECTOR *scaler = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(scaler_f));
  VECTOR *frc = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(frc_f));
  restrainforcekernel<<<block_per_grid, thread_per_block, 0, stream>>>(restrain_numbers, restrain_list, uint_crd,
                                                                       uint_crd_ref, factor, scaler, frc);
  return;
}

void restrainforce(int restrain_numbers, int atom_numbers, const int *restrain_list, const int *uint_crd_f,
                   const int *uint_crd_ref, const float factor, const float *scaler_f, float *frc_f,
                   cudaStream_t stream);

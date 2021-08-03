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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/restrain/restrain_energy_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

__global__ void restrain_energy_kernel(const int restrain_numbers, const int *restrain_list, const VECTOR *crd,
                                       const VECTOR *crd_ref, const float weight, const VECTOR boxlength, float *ene) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < restrain_numbers) {
    int atom_i = restrain_list[i];
    VECTOR dr = Get_Periodic_Displacement(crd_ref[atom_i], crd[atom_i], boxlength);
    ene[i] = weight * dr * dr;
  }
}

void restrainenergy(int restrain_numbers, int atom_numbers, float weight, const int *restrain_list, const float *crd_f,
                    const float *crd_ref_f, const float *boxlength_f, float *ene, cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(restrain_numbers) / 128), 128, 0, stream>>>(restrain_numbers, ene, 0.);
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(restrain_numbers) / 128);
  const VECTOR *crd = reinterpret_cast<const VECTOR *>(crd_f);
  const VECTOR *crd_ref = reinterpret_cast<const VECTOR *>(crd_ref_f);
  const VECTOR *boxlength = reinterpret_cast<const VECTOR *>(boxlength_f);
  restrain_energy_kernel<<<block_per_grid, thread_per_block, 0, stream>>>(restrain_numbers, restrain_list, crd, crd_ref,
                                                                          weight, boxlength[0], ene);
  return;
}

void restrainenergy(int restrain_numbers, int atom_numbers, float weight, const int *restrain_list, const float *crd_f,
                    const float *crd_ref, const float *boxlength_f, float *ene, cudaStream_t stream);

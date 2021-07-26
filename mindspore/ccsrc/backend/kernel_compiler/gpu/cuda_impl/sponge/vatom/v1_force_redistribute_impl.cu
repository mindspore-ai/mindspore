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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/vatom/v1_force_redistribute_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

__global__ void v1_Force_Redistribute(const int virtual_numbers, const VIRTUAL_TYPE_1 *v_info,
                                      const UNSIGNED_INT_VECTOR *uint_crd, VECTOR *force) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < virtual_numbers) {
    VIRTUAL_TYPE_1 v_temp = v_info[i];
    int atom_v = v_temp.virtual_atom;
    int atom_1 = v_temp.from_1;
    int atom_2 = v_temp.from_2;
    float a = v_temp.a;
    VECTOR force_v = force[atom_v];
    atomicAdd(&force[atom_1].x, a * force_v.x);
    atomicAdd(&force[atom_1].y, a * force_v.y);
    atomicAdd(&force[atom_1].z, a * force_v.z);

    atomicAdd(&force[atom_2].x, (1 - a) * force_v.x);
    atomicAdd(&force[atom_2].y, (1 - a) * force_v.y);
    atomicAdd(&force[atom_2].z, (1 - a) * force_v.z);

    force_v.x = 0.0f;
    force_v.y = 0.0f;
    force_v.z = 0.0f;
    force[atom_v] = force_v;
  }
}

void v1ForceRedistribute(int atom_numbers, int virtual_numbers, const float *v_info_f, const int *uint_crd_f,
                         float *frc_f, cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, frc_f, 0.);
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(virtual_numbers) / 128);
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));

  VECTOR *frc = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(frc_f));
  VIRTUAL_TYPE_1 *v_info = const_cast<VIRTUAL_TYPE_1 *>(reinterpret_cast<const VIRTUAL_TYPE_1 *>(v_info_f));

  v1_Force_Redistribute<<<block_per_grid, thread_per_block, 0, stream>>>(virtual_numbers, v_info, uint_crd, frc);

  return;
}

void v1ForceRedistribute(int atom_numbers, int virtual_numbers, const float *v_info_f, const int *uint_crd_f,
                         float *frc_f, cudaStream_t stream);

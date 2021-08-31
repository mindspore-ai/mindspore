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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common/total_c6_get_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void Total_C6_Get(int atom_numbers, int *atom_lj_type, float *d_lj_b, float *d_factor) {
  int i, j;
  float temp_sum = 0;
  d_factor[0] = 0;
  int x, y;
  int itype, jtype, atom_pair_LJ_type;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i += gridDim.x * blockDim.x) {
    itype = atom_lj_type[i];
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < atom_numbers; j += gridDim.y * blockDim.y) {
      jtype = atom_lj_type[j];
      y = (jtype - itype);
      x = y >> 31;
      y = (y ^ x) - x;
      x = jtype + itype;
      jtype = (x + y) >> 1;
      x = (x - y) >> 1;
      atom_pair_LJ_type = (jtype * (jtype + 1) >> 1) + x;
      temp_sum += d_lj_b[atom_pair_LJ_type];
    }
  }
  atomicAdd(d_factor, temp_sum);
}

void total_c6_get(int atom_numbers, int *atom_lj_type, float *d_lj_b, float *d_factor, cudaStream_t stream) {
  Total_C6_Get<<<{4, 4}, {32, 32}, 0, stream>>>(atom_numbers, atom_lj_type, d_lj_b, d_factor);
  return;
}

void total_c6_get(int atom_numbers, int *atom_lj_type, float *d_lj_b, float *d_factor, cudaStream_t stream);

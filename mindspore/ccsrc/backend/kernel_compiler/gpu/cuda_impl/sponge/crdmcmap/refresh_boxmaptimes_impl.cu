
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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/crdmcmap/refresh_boxmaptimes_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void Refresh_BoxMapTimes_CUDA(int atom_numbers, VECTOR *box_length_inverse, VECTOR *crd,
                                         INT_VECTOR *box_map_times, VECTOR *old_crd) {
  VECTOR crd_i, old_crd_i;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i += gridDim.x * blockDim.x) {
    crd_i = crd[i];
    old_crd_i = old_crd[i];
    box_map_times[i].int_x += floor((old_crd_i.x - crd_i.x) * box_length_inverse[0].x + 0.5);
    box_map_times[i].int_y += floor((old_crd_i.y - crd_i.y) * box_length_inverse[0].y + 0.5);
    box_map_times[i].int_z += floor((old_crd_i.z - crd_i.z) * box_length_inverse[0].z + 0.5);
    old_crd[i] = crd_i;
  }
}

void refresh_boxmaptimes(int atom_numbers, float *box_length_inverse_f, float *crd_f, float *old_crd_f,
                         int *box_map_times_f, cudaStream_t stream) {
  INT_VECTOR *box_map_times = reinterpret_cast<INT_VECTOR *>(box_map_times_f);
  VECTOR *box_length_inverse = reinterpret_cast<VECTOR *>(box_length_inverse_f);
  VECTOR *crd = reinterpret_cast<VECTOR *>(crd_f);
  VECTOR *old_crd = reinterpret_cast<VECTOR *>(old_crd_f);

  Refresh_BoxMapTimes_CUDA<<<1, 256, 0, stream>>>(atom_numbers, box_length_inverse, crd,
                                                                            box_map_times, old_crd);
  return;
}

void refresh_boxmaptimes(int atom_numbers, float *box_length_inverse, float *crd_f, float *old_crd_f,
                         int *box_map_times_f, cudaStream_t stream);

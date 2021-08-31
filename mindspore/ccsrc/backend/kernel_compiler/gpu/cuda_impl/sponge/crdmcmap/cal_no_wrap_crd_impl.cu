
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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/crdmcmap/cal_no_wrap_crd_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

__global__ void Calculate_No_Wrap_Crd(int atom_numbers, INT_VECTOR *box_map_times, VECTOR *box, VECTOR *crd,
                                      VECTOR *nowrap_crd) {
  for (int i = threadIdx.x; i < atom_numbers; i = i + blockDim.x) {
    nowrap_crd[i].x = static_cast<float>(box_map_times[i].int_x) * box[0].x + crd[i].x;
    nowrap_crd[i].y = static_cast<float>(box_map_times[i].int_y) * box[0].y + crd[i].y;
    nowrap_crd[i].z = static_cast<float>(box_map_times[i].int_z) * box[0].z + crd[i].z;
  }
}

void calculatenowrapcrd(int atom_numbers, int *box_map_times_f, float *box_f, float *crd_f, float *nowrap_crd_f,
                        cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, nowrap_crd_f,
                                                                                     0.);
  INT_VECTOR *box_map_times = reinterpret_cast<INT_VECTOR *>(box_map_times_f);
  VECTOR *box = reinterpret_cast<VECTOR *>(box_f);
  VECTOR *crd = reinterpret_cast<VECTOR *>(crd_f);
  VECTOR *nowrap_crd = reinterpret_cast<VECTOR *>(nowrap_crd_f);

  Calculate_No_Wrap_Crd<<<20, 256, 0, stream>>>(atom_numbers, box_map_times, box, crd,
                                                                         nowrap_crd);
  return;
}

void calculatenowrapcrd(int atom_numbers, int *box_map_times_f, float *box_f, float *crd_f, float *nowrap_crd_f,
                        cudaStream_t stream);

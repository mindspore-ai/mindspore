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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common/crd_to_uint_crd_impl.cuh"

__global__ void Crd_To_Uint_Crd(const int atom_numbers, const VECTOR *scale_factor, const VECTOR *crd,
                                UNSIGNED_INT_VECTOR *uint_crd) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    uint_crd[atom_i].uint_x = crd[atom_i].x * scale_factor[0].x;
    uint_crd[atom_i].uint_y = crd[atom_i].y * scale_factor[0].y;
    uint_crd[atom_i].uint_z = crd[atom_i].z * scale_factor[0].z;
    /*uint_crd[atom_i].uint_x = 2 * uint_crd[atom_i].uint_x;
    uint_crd[atom_i].uint_y = 2 * uint_crd[atom_i].uint_y;
    uint_crd[atom_i].uint_z = 2 * uint_crd[atom_i].uint_z;*/
    uint_crd[atom_i].uint_x = uint_crd[atom_i].uint_x << 1;
    uint_crd[atom_i].uint_y = uint_crd[atom_i].uint_y << 1;
    uint_crd[atom_i].uint_z = uint_crd[atom_i].uint_z << 1;
  }
}

void CrdToUintCrd(const int atom_numbers, const float *crd_to_uint_crd_cof_f, const float *crd_f,
                  unsigned int *uint_crd_f, cudaStream_t stream) {
  VECTOR *crd = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(crd_f));
  VECTOR *crd_to_uint_crd_cof = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(crd_to_uint_crd_cof_f));

  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));

  Crd_To_Uint_Crd<<<ceilf(static_cast<float>(atom_numbers) / 128.0), 128, 0, stream>>>(
    atom_numbers, crd_to_uint_crd_cof, crd, uint_crd);

  return;
}

void CrdToUintCrd(const int atom_numbers, const float *crd_to_uint_crd_cof_f, const float *crd_f,
                  unsigned int *uint_crd_f, cudaStream_t stream);

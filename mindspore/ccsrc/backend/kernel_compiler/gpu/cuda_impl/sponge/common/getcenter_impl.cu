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
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common/getcenter_impl.cuh"

__global__ void GetCenterOfGeometryKernel(const int center_numbers, float center_numbers_inverse,
                                          const int *center_atoms, const VECTOR *crd, VECTOR *center_of_geometry) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < center_numbers) {
    int atom_i = center_atoms[i];
    VECTOR temp = center_numbers_inverse * crd[atom_i];
    atomicAdd(&center_of_geometry[0].x, temp.x);
    atomicAdd(&center_of_geometry[0].y, temp.y);
    atomicAdd(&center_of_geometry[0].z, temp.z);
  }
}

void GetCenterOfGeometry(const int center_numbers, float center_numbers_inverse, const int *center_atoms,
                         const float *crd_f, float *center_of_geometry_f, cudaStream_t stream) {
  VECTOR *crd = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(crd_f));
  VECTOR *center_of_geometry = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(center_of_geometry_f));
  GetCenterOfGeometryKernel<<<ceilf(static_cast<float>(center_numbers) / 32), 32, 0, stream>>>(
    center_numbers, center_numbers_inverse, center_atoms, crd, center_of_geometry);

  cudaStreamSynchronize(stream);

  return;
}

void GetCenterOfGeometry(const int center_numbers, float center_numbers_inverse, const int *center_atoms, float *crd_f,
                         float *center_of_geometry_f, cudaStream_t stream);

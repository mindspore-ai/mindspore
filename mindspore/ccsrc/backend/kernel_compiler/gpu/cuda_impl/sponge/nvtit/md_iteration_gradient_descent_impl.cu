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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/nvtit/md_iteration_gradient_descent_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void MD_Iteration_Gradient_Descent(const int atom_numbers, VECTOR *crd, VECTOR *frc,
                                              const float learning_rate) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < atom_numbers) {
    crd[i].x = crd[i].x + learning_rate * frc[i].x;
    crd[i].y = crd[i].y + learning_rate * frc[i].y;
    crd[i].z = crd[i].z + learning_rate * frc[i].z;

    frc[i].x = 0.;
    frc[i].y = 0.;
    frc[i].z = 0.;
  }
}

void MDIterationGradientDescent(const int atom_numbers, float *crd, float *frc, const float learning_rate,
                                cudaStream_t stream) {
  VECTOR *d_crd = reinterpret_cast<VECTOR *>(crd);
  VECTOR *d_frc = reinterpret_cast<VECTOR *>(frc);
  MD_Iteration_Gradient_Descent<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(
    atom_numbers, d_crd, d_frc, learning_rate);
}

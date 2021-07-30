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
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common/get_center_of_mass_impl.cuh"

__global__ void Get_Center_Of_Mass(int residue_numbers, int *start, int *end, VECTOR *crd, float *atom_mass,
                                   float *residue_mass_inverse, VECTOR *center_of_mass) {
  for (int residue_i = blockDim.x * blockIdx.x + threadIdx.x; residue_i < residue_numbers;
       residue_i += gridDim.x * blockDim.x) {
    VECTOR com_lin = {0.0f, 0.0f, 0.0f};
    for (int atom_i = start[residue_i]; atom_i < end[residue_i]; atom_i += 1) {
      com_lin = com_lin + atom_mass[atom_i] * crd[atom_i];
    }
    center_of_mass[residue_i] = residue_mass_inverse[residue_i] * com_lin;
  }
}

void GetCenterOfMass(int residue_numbers, int *start, int *end, float *crd_f, float *atom_mass,
                     float *residue_mass_inverse, float *center_of_mass_f, cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * residue_numbers) / 128), 128, 0, stream>>>(3 * residue_numbers,
                                                                                        center_of_mass_f, 0.);
  VECTOR *crd = reinterpret_cast<VECTOR *>(crd_f);
  VECTOR *center_of_mass = reinterpret_cast<VECTOR *>(center_of_mass_f);
  Get_Center_Of_Mass<<<20, 32, 0, stream>>>(residue_numbers, start, end, crd, atom_mass, residue_mass_inverse,
                                            center_of_mass);
  return;
}

void GetCenterOfMass(int residue_numbers, int *start, int *end, float *crd_f, float *atom_mass,
                     float *residue_mass_inverse, float *center_of_mass_f, cudaStream_t stream);

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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/angle/angle_atom_energy_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void AngleAtomEnergyKernel(int angle_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR *scaler,
                                      const int *atom_a, const int *atom_b, const int *atom_c, const float *angle_k,
                                      const float *angle_theta0, float *atom_energy) {
  int angle_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (angle_i < angle_numbers) {
    int atom_i = atom_a[angle_i];
    int atom_j = atom_b[angle_i];
    int atom_k = atom_c[angle_i];

    float theta0 = angle_theta0[angle_i];
    float k = angle_k[angle_i];

    VECTOR drij = Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler[0]);
    VECTOR drkj = Get_Periodic_Displacement(uint_crd[atom_k], uint_crd[atom_j], scaler[0]);

    float rij_2 = 1. / (drij * drij);
    float rkj_2 = 1. / (drkj * drkj);
    float rij_1_rkj_1 = sqrtf(rij_2 * rkj_2);

    float costheta = drij * drkj * rij_1_rkj_1;
    costheta = fmaxf(-0.999999, fminf(costheta, 0.999999));
    float theta = acosf(costheta);

    float dtheta = theta - theta0;

    atomicAdd(&atom_energy[atom_i], k * dtheta * dtheta);
  }
}

void AngleAtomEnergy(int angle_numbers, int atom_numbers, const int *uint_crd_f, const float *scaler_f,
                     const int *atom_a, const int *atom_b, const int *atom_c, const float *angle_k,
                     const float *angle_theta0, float *ene, cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(atom_numbers, ene, 0.);
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(angle_numbers) / 128);
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));
  VECTOR *scaler = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(scaler_f));

  AngleAtomEnergyKernel<<<block_per_grid, thread_per_block, 0, stream>>>(angle_numbers, uint_crd, scaler, atom_a,
                                                                         atom_b, atom_c, angle_k, angle_theta0, ene);
  return;
}
void AngleAtomEnergy(int angle_numbers, int atom_numbers, const int *uint_crd_f, const float *scaler_f,
                     const int *atom_a, const int *atom_b, const int *atom_c, const float *angle_k,
                     const float *angle_theta0, float *ene, cudaStream_t stream);

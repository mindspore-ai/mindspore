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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/bond/bond_force_cuda_gpu_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

__global__ void BondForceCudaKernel(int bond_numbers, const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR *scaler,
                                    const int *atom_a, const int *atom_b, const float *bond_k, const float *bond_r0,
                                    VECTOR *frc) {
  int bond_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (bond_i < bond_numbers) {
    int atom_i = atom_a[bond_i];
    int atom_j = atom_b[bond_i];

    float k = bond_k[bond_i];
    float r0 = bond_r0[bond_i];
    VECTOR dr = Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler[0]);
    float r_1 = rnorm3df(dr.x, dr.y, dr.z);
    float tempf = 1.0 - r0 * r_1;

    VECTOR f = 2 * tempf * k * dr;
    atomicAdd(&frc[atom_i].x, -f.x);
    atomicAdd(&frc[atom_i].y, -f.y);
    atomicAdd(&frc[atom_i].z, -f.z);

    atomicAdd(&frc[atom_j].x, f.x);
    atomicAdd(&frc[atom_j].y, f.y);
    atomicAdd(&frc[atom_j].z, f.z);
  }
}

void BondForce(int bond_numbers, int atom_numbers, const int *uint_crd_f, const float *scaler_f, const int *atom_a,
               const int *atom_b, const float *bond_k, const float *bond_r0, float *frc_f, cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, frc_f, 0.);
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(bond_numbers) / 128);
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));
  VECTOR *scaler = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(scaler_f));
  VECTOR *frc = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(frc_f));
  BondForceCudaKernel<<<block_per_grid, thread_per_block, 0, stream>>>(bond_numbers, uint_crd, scaler, atom_a, atom_b,
                                                                       bond_k, bond_r0, frc);
  return;
}

void BondForce(int bond_numbers, int atom_numbers, const int *uint_crd_f, const float *scaler_f, const int *atom_a,
               const int *atom_b, const float *bond_k, const float *bond_r0, float *frc_f, cudaStream_t stream);

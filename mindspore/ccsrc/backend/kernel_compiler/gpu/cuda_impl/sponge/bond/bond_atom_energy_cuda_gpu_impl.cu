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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/bond/bond_atom_energy_cuda_gpu_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

__global__ void BondAtomEnergyCudaKernel(const int bond_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
                                         const VECTOR *scaler, const int *atom_a, const int *atom_b,
                                         const float *bond_k, const float *bond_r0, float *atom_ene) {
  int bond_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (bond_i < bond_numbers) {
    int atom_i = atom_a[bond_i];
    int atom_j = atom_b[bond_i];

    float k = bond_k[bond_i];
    float r0 = bond_r0[bond_i];

    VECTOR dr = Get_Periodic_Displacement(uint_crd[atom_i], uint_crd[atom_j], scaler[0]);

    float r1 = norm3df(dr.x, dr.y, dr.z);
    float tempf = r1 - r0;

    atomicAdd(&atom_ene[atom_i], k * tempf * tempf);
  }
}

void BondAtomEnergy(int bond_numbers, int atom_numbers, const int *uint_crd_f, const float *scaler_f, const int *atom_a,
                    const int *atom_b, const float *bond_k, const float *bond_r0, float *atom_ene,
                    cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(atom_numbers, atom_ene, 0.);
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(bond_numbers) / 128);
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));
  VECTOR *scaler = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(scaler_f));

  BondAtomEnergyCudaKernel<<<block_per_grid, thread_per_block, 0, stream>>>(bond_numbers, uint_crd, scaler, atom_a,
                                                                            atom_b, bond_k, bond_r0, atom_ene);
  return;
}

void BondAtomEnergy(int bond_numbers, int atom_numbers, const int *uint_crd_f, const float *scaler_f, const int *atom_a,
                    const int *atom_b, const float *bond_k, const float *bond_r0, float *atom_ene, cudaStream_t stream);

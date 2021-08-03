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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/restrain/restrain_force_atom_energy_virial_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void restrain_force_with_atom_energy_and_virial(int restrain_numbers, const int *restrain_list,
                                                           const VECTOR *crd, const VECTOR *crd_ref, const float weight,
                                                           const VECTOR *boxlength, float *atom_energy,
                                                           float *atom_virial, VECTOR *frc) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < restrain_numbers) {
    int atom_i = restrain_list[i];
    VECTOR dr = Get_Periodic_Displacement(crd_ref[atom_i], crd[atom_i], boxlength[0]);

    VECTOR temp_force = 2 * weight * dr;
    float virial = temp_force * dr;

    atom_energy[atom_i] += 0.5 * virial;
    atom_virial[atom_i] -= virial;
    frc[atom_i] = frc[atom_i] + temp_force;
  }
}

void restrainforcewithatomenergyandvirial(int restrain_numbers, int atom_numbers, const int *restrain_list,
                                          const float *crd_f, const float *crd_ref_f, const float weight,
                                          const float *boxlength_f, float *atom_ene, float *atom_virial, float *frc_f,
                                          cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, frc_f, 0.);
  Reset_List<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(atom_numbers, atom_ene, 0.);
  Reset_List<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(atom_numbers, atom_virial, 0.);
  size_t thread_per_block = 128;
  size_t block_per_grid = ceilf(static_cast<float>(restrain_numbers) / 128);
  const VECTOR *crd = reinterpret_cast<const VECTOR *>(crd_f);
  const VECTOR *crd_ref = reinterpret_cast<const VECTOR *>(crd_ref_f);
  const VECTOR *boxlength = reinterpret_cast<const VECTOR *>(boxlength_f);
  VECTOR *frc = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(frc_f));
  restrain_force_with_atom_energy_and_virial<<<block_per_grid, thread_per_block, 0, stream>>>(
    restrain_numbers, restrain_list, crd, crd_ref, weight, boxlength, atom_ene, atom_virial, frc);
  return;
}

void restrainforcewithatomenergyandvirial(int restrain_numbers, int atom_numbers, const int *restrain_list,
                                          const float *crd_f, const float *crd_ref_f, const float weight,
                                          const float *boxlength_f, float *atom_ene, float *atom_virial, float *frc_f,
                                          cudaStream_t stream);

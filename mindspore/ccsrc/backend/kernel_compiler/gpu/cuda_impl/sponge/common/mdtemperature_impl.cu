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
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common/mdtemperature_impl.cuh"

__global__ void MDTemperatureKernel(const int residue_numbers, const int *start, const int *end, const VECTOR *atom_vel,
                                    const float *atom_mass, float *ek) {
  int residue_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (residue_i < residue_numbers) {
    VECTOR momentum = {0., 0., 0.};
    float res_mass = 0.;
    int s = start[residue_i];
    int e = end[residue_i];
    float mass_lin;
    for (int atom_i = s; atom_i < e; atom_i = atom_i + 1) {
      mass_lin = atom_mass[atom_i];

      momentum.x = momentum.x + mass_lin * atom_vel[atom_i].x;
      momentum.y = momentum.y + mass_lin * atom_vel[atom_i].y;
      momentum.z = momentum.z + mass_lin * atom_vel[atom_i].z;
      res_mass = res_mass + mass_lin;
    }
    ek[residue_i] = 0.5 * (momentum.x * momentum.x + momentum.y * momentum.y + momentum.z * momentum.z) / res_mass *
                    2. / 3. / CONSTANT_kB / residue_numbers;
  }
}

void MDTemperature(const int residue_numbers, const int *start, const int *end, const float *atom_vel_f,
                   const float *atom_mass, float *ek, cudaStream_t stream) {
  VECTOR *atom_vel = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(atom_vel_f));
  MDTemperatureKernel<<<ceilf(static_cast<float>(residue_numbers) / 32), 32, 0, stream>>>(residue_numbers, start, end,
                                                                                          atom_vel, atom_mass, ek);
  cudaStreamSynchronize(stream);

  return;
}
void MDTemperature(const int residue_numbers, const int *start, const int *end, const float *atom_vel_f,
                   const float *atom_mass, float *ek, cudaStream_t stream);

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
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/pme_excluded_force_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/pme_common.cuh"

__global__ void PME_Excluded_Force_Correction(const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
                                              const VECTOR *sacler, const float *charge, const float pme_beta,
                                              const float sqrt_pi, const int *excluded_list_start,
                                              const int *excluded_list, const int *excluded_atom_numbers, VECTOR *frc) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    int excluded_numbers = excluded_atom_numbers[atom_i];
    if (excluded_numbers > 0) {
      int list_start = excluded_list_start[atom_i];
      // int atom_min = excluded_list[list_start];
      int list_end = list_start + excluded_numbers;
      int atom_j;
      int int_x;
      int int_y;
      int int_z;

      float charge_i = charge[atom_i];
      float charge_j;
      float dr_abs;
      float beta_dr;

      UNSIGNED_INT_VECTOR r1 = uint_crd[atom_i], r2;
      VECTOR dr;
      float dr2;

      float frc_abs = 0.;
      VECTOR frc_lin;
      VECTOR frc_record = {0., 0., 0.};

      for (int i = list_start; i < list_end; i = i + 1) {
        atom_j = excluded_list[i];
        r2 = uint_crd[atom_j];
        charge_j = charge[atom_j];

        int_x = r2.uint_x - r1.uint_x;
        int_y = r2.uint_y - r1.uint_y;
        int_z = r2.uint_z - r1.uint_z;
        dr.x = sacler[0].x * int_x;
        dr.y = sacler[0].y * int_y;
        dr.z = sacler[0].z * int_z;
        dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

        dr_abs = sqrtf(dr2);
        beta_dr = pme_beta * dr_abs;
        // sqrt_pi= 2/sqrt(3.141592654);
        frc_abs = beta_dr * sqrt_pi * expf(-beta_dr * beta_dr) + erfcf(beta_dr);
        frc_abs = (frc_abs - 1.) / dr2 / dr_abs;
        frc_abs = -charge_i * charge_j * frc_abs;
        frc_lin.x = frc_abs * dr.x;
        frc_lin.y = frc_abs * dr.y;
        frc_lin.z = frc_abs * dr.z;

        frc_record.x = frc_record.x + frc_lin.x;
        frc_record.y = frc_record.y + frc_lin.y;
        frc_record.z = frc_record.z + frc_lin.z;

        atomicAdd(&frc[atom_j].x, -frc_lin.x);
        atomicAdd(&frc[atom_j].y, -frc_lin.y);
        atomicAdd(&frc[atom_j].z, -frc_lin.z);
      }  // atom_j cycle
      atomicAdd(&frc[atom_i].x, frc_record.x);
      atomicAdd(&frc[atom_i].y, frc_record.y);
      atomicAdd(&frc[atom_i].z, frc_record.z);
    }  // if need excluded
  }
}

void PMEExcludedForce(const int atom_numbers, const float pme_beta, const int *uint_crd_f, const float *sacler_f,
                      const float *charge, const int *excluded_list_start, const int *excluded_list,
                      const int *excluded_atom_numbers, float *frc_f, cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, frc_f, 0.);
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));
  VECTOR *frc = reinterpret_cast<VECTOR *>(frc_f);
  VECTOR *sacler = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(sacler_f));

  PME_Excluded_Force_Correction<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(
    atom_numbers, uint_crd, sacler, charge, pme_beta, TWO_DIVIDED_BY_SQRT_PI, excluded_list_start, excluded_list,
    excluded_atom_numbers, frc);
  return;
}

void PMEExcludedForce(const int atom_numbers, const float pme_beta, const int *uint_crd_f, const float *sacler_f,
                      const float *charge, const int *excluded_list_start, const int *excluded_list,
                      const int *excluded_atom_numbers, float *frc_f, cudaStream_t stream);

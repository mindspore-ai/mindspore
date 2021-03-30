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
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/pme_energy_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/pme_common.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"

__global__ void PME_Energy_Product(const int element_number, const float *list1, const float *list2, float *sum) {
  if (threadIdx.x == 0) {
    sum[0] = 0.;
  }
  __syncthreads();
  float lin = 0.0;
  for (int i = threadIdx.x; i < element_number; i = i + blockDim.x) {
    lin = lin + list1[i] * list2[i];
  }
  atomicAdd(sum, lin);
}

__global__ void PME_Energy_Reciprocal(const int element_number, const cufftComplex *FQ, const float *BC, float *sum) {
  if (threadIdx.x == 0) {
    sum[0] = 0.;
  }
  __syncthreads();
  float lin = 0.0;
  cufftComplex FQ_i;
  for (int i = threadIdx.x; i < element_number; i = i + blockDim.x) {
    FQ_i = FQ[i];
    lin = lin + (FQ_i.x * FQ_i.x + FQ_i.y * FQ_i.y) * BC[i];
  }
  atomicAdd(sum, lin);
}

__global__ void PME_Excluded_Energy_Correction(const int atom_numbers, const UNSIGNED_INT_VECTOR *uint_crd,
                                               const VECTOR *sacler, const float *charge, const float pme_beta,
                                               const float sqrt_pi, const int *excluded_list_start,
                                               const int *excluded_list, const int *excluded_atom_numbers, float *ene) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    int excluded_number = excluded_atom_numbers[atom_i];
    if (excluded_number > 0) {
      int list_start = excluded_list_start[atom_i];
      // int atom_min = excluded_list[list_start];
      int list_end = list_start + excluded_number;
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

      float ene_lin = 0.;

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

        ene_lin -= charge_i * charge_j * erff(beta_dr) / dr_abs;
      }
      atomicAdd(ene, ene_lin);
    }
  }
}

void PMEEnergy(int fftx, int ffty, int fftz, int atom_numbers, float beta, float *PME_BC, int *pme_uxyz,
               float *pme_frxyz, float *PME_Q, float *pme_fq, int *PME_atom_near, int *pme_kxyz, const int *uint_crd_f,
               const float *charge, int *nl_atom_numbers, int *nl_atom_serial, int *nl, const float *scaler_f,
               const int *excluded_list_start, const int *excluded_list, const int *excluded_atom_numbers,
               float *d_reciprocal_ene, float *d_self_ene, float *d_direct_ene, float *d_correction_ene,
               dim3 thread_PME, int PME_Nin, int PME_Nfft, int PME_Nall, const cufftHandle &PME_plan_r2c,
               const cufftHandle &PME_plan_c2r, cudaStream_t stream) {
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));
  VECTOR *scaler = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(scaler_f));
  int max_neighbor_numbers = 800;
  NEIGHBOR_LIST *nl_a = reinterpret_cast<NEIGHBOR_LIST *>(nl);
  construct_neighbor_list_kernel<<<ceilf(static_cast<float>(atom_numbers) / 128), 128, 0, stream>>>(
    atom_numbers, max_neighbor_numbers, nl_atom_numbers, nl_atom_serial, nl_a);

  UNSIGNED_INT_VECTOR *PME_uxyz = reinterpret_cast<UNSIGNED_INT_VECTOR *>(pme_uxyz);
  UNSIGNED_INT_VECTOR *PME_kxyz = reinterpret_cast<UNSIGNED_INT_VECTOR *>(pme_kxyz);
  VECTOR *PME_frxyz = reinterpret_cast<VECTOR *>(pme_frxyz);
  cufftComplex *PME_FQ = reinterpret_cast<cufftComplex *>(pme_fq);

  Reset_List<<<3 * atom_numbers / 32 + 1, 32, 0, stream>>>(3 * atom_numbers, reinterpret_cast<int *>(PME_uxyz),
                                                           1 << 30);
  PME_Atom_Near<<<atom_numbers / 32 + 1, 32, 0, stream>>>(
    uint_crd, PME_atom_near, PME_Nin, periodic_factor_inverse * fftx, periodic_factor_inverse * ffty,
    periodic_factor_inverse * fftz, atom_numbers, fftx, ffty, fftz, PME_kxyz, PME_uxyz, PME_frxyz);

  Reset_List<<<PME_Nall / 1024 + 1, 1024, 0, stream>>>(PME_Nall, PME_Q, 0);

  PME_Q_Spread<<<atom_numbers / thread_PME.x + 1, thread_PME, 0, stream>>>(PME_atom_near, charge, PME_frxyz, PME_Q,
                                                                           PME_kxyz, atom_numbers);

  cufftExecR2C(PME_plan_r2c, reinterpret_cast<float *>(PME_Q), reinterpret_cast<cufftComplex *>(PME_FQ));

  PME_Energy_Reciprocal<<<1, 1024, 0, stream>>>(PME_Nfft, PME_FQ, PME_BC, d_reciprocal_ene);

  PME_Energy_Product<<<1, 1024, 0, stream>>>(atom_numbers, charge, charge, d_self_ene);
  Scale_List<<<1, 1, 0, stream>>>(1, d_self_ene, -beta / sqrtf(PI));

  Reset_List<<<1, 1, 0, stream>>>(1, d_direct_ene, 0.0);
  PME_Direct_Energy<<<atom_numbers / thread_PME.x + 1, thread_PME, 0, stream>>>(
    atom_numbers, nl_a, uint_crd, scaler, charge, beta, cutoff * cutoff, d_direct_ene);

  Reset_List<<<1, 1, 0, stream>>>(1, d_correction_ene, 0.0);
  PME_Excluded_Energy_Correction<<<atom_numbers / 32 + 1, 32, 0, stream>>>(
    atom_numbers, uint_crd, scaler, charge, beta, sqrtf(PI), excluded_list_start, excluded_list, excluded_atom_numbers,
    d_correction_ene);
  return;
}

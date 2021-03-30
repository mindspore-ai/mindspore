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
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/pme_reciprocal_force_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/pme_common.cuh"

__global__ void PME_BCFQ(cufftComplex *PME_FQ, float *PME_BC, int PME_Nfft) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index < PME_Nfft) {
    float tempf = PME_BC[index];
    cufftComplex tempc = PME_FQ[index];
    PME_FQ[index].x = tempc.x * tempf;
    PME_FQ[index].y = tempc.y * tempf;
  }
}

__global__ void PME_Final(int *PME_atom_near, const float *charge, const float *PME_Q, VECTOR *force,
                          const VECTOR *PME_frxyz, const UNSIGNED_INT_VECTOR *PME_kxyz,
                          const _VECTOR PME_inverse_box_vector, const int atom_numbers) {
  int atom = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom < atom_numbers) {
    int k, kx;
    float tempdQx, tempdQy, tempdQz, tempdx, tempdy, tempdz, tempx, tempy, tempz, tempdQf;
    float tempf, tempf2;
    float temp_charge = charge[atom];
    int *temp_near = PME_atom_near + atom * 64;
    UNSIGNED_INT_VECTOR temp_kxyz;
    VECTOR temp_frxyz = PME_frxyz[atom];
    for (k = threadIdx.y; k < 64; k = k + blockDim.y) {
      temp_kxyz = PME_kxyz[k];
      tempdQf = -PME_Q[temp_near[k]] * temp_charge;

      kx = temp_kxyz.uint_x;
      tempf = (temp_frxyz.x);
      tempf2 = tempf * tempf;
      tempx = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];
      tempdx = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

      kx = temp_kxyz.uint_y;
      tempf = (temp_frxyz.y);
      tempf2 = tempf * tempf;
      tempy = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];
      tempdy = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

      kx = temp_kxyz.uint_z;
      tempf = (temp_frxyz.z);
      tempf2 = tempf * tempf;
      tempz = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];
      tempdz = PME_dMa[kx] * tempf2 + PME_dMb[kx] * tempf + PME_dMc[kx];

      tempdQx = tempdx * tempy * tempz * PME_inverse_box_vector.x;
      tempdQy = tempdy * tempx * tempz * PME_inverse_box_vector.y;
      tempdQz = tempdz * tempx * tempy * PME_inverse_box_vector.z;

      atomicAdd(&force[atom].x, tempdQf * tempdQx);
      atomicAdd(&force[atom].y, tempdQf * tempdQy);
      atomicAdd(&force[atom].z, tempdQf * tempdQz);
    }
  }
}

void PMEReciprocalForce(int fftx, int ffty, int fftz, int atom_numbers, float beta, float *PME_BC, int *pme_uxyz,
                        float *pme_frxyz, float *PME_Q, float *pme_fq, int *PME_atom_near, int *pme_kxyz,
                        const int *uint_crd_f, const float *charge, float *force, int PME_Nin, int PME_Nall,
                        int PME_Nfft, const cufftHandle &PME_plan_r2c, const cufftHandle &PME_plan_c2r,
                        const _VECTOR &PME_inverse_box_vector, cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, force, 0.);
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));
  UNSIGNED_INT_VECTOR *PME_uxyz = reinterpret_cast<UNSIGNED_INT_VECTOR *>(pme_uxyz);
  UNSIGNED_INT_VECTOR *PME_kxyz = reinterpret_cast<UNSIGNED_INT_VECTOR *>(pme_kxyz);
  Reset_List<<<3 * atom_numbers / 32 + 1, 32, 0, stream>>>(3 * atom_numbers, reinterpret_cast<int *>(PME_uxyz),
                                                           1 << 30);

  VECTOR *PME_frxyz = reinterpret_cast<VECTOR *>(pme_frxyz);
  VECTOR *frc = reinterpret_cast<VECTOR *>(force);

  cufftComplex *PME_FQ = reinterpret_cast<cufftComplex *>(pme_fq);

  // initial end
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(
    3 * atom_numbers, reinterpret_cast<float *>(frc), 0.);
  PME_Atom_Near<<<atom_numbers / 32 + 1, 32, 0, stream>>>(
    uint_crd, PME_atom_near, PME_Nin, periodic_factor_inverse * fftx, periodic_factor_inverse * ffty,
    periodic_factor_inverse * fftz, atom_numbers, fftx, ffty, fftz, PME_kxyz, PME_uxyz, PME_frxyz);
  Reset_List<<<PME_Nall / 1024 + 1, 1024, 0, stream>>>(PME_Nall, PME_Q, 0);

  PME_Q_Spread<<<atom_numbers / thread_PME.x + 1, thread_PME, 0, stream>>>(PME_atom_near, charge, PME_frxyz, PME_Q,
                                                                           PME_kxyz, atom_numbers);

  cufftExecR2C(PME_plan_r2c, reinterpret_cast<float *>(PME_Q), reinterpret_cast<cufftComplex *>(PME_FQ));
  PME_BCFQ<<<PME_Nfft / 1024 + 1, 1024, 0, stream>>>(PME_FQ, PME_BC, PME_Nfft);

  cufftExecC2R(PME_plan_c2r, reinterpret_cast<cufftComplex *>(PME_FQ), reinterpret_cast<float *>(PME_Q));

  PME_Final<<<atom_numbers / thread_PME.x + 1, thread_PME, 0, stream>>>(PME_atom_near, charge, PME_Q, frc, PME_frxyz,
                                                                        PME_kxyz, PME_inverse_box_vector, atom_numbers);
  return;
}

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
                          const VECTOR PME_inverse_box_vector, const int atom_numbers) {
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
                        const float *box_length_f, const int *uint_crd_f, const float *charge, float *force,
                        cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, force, 0.);
  UNSIGNED_INT_VECTOR *uint_crd =
    const_cast<UNSIGNED_INT_VECTOR *>(reinterpret_cast<const UNSIGNED_INT_VECTOR *>(uint_crd_f));
  UNSIGNED_INT_VECTOR *PME_uxyz = reinterpret_cast<UNSIGNED_INT_VECTOR *>(pme_uxyz);
  UNSIGNED_INT_VECTOR *PME_kxyz = reinterpret_cast<UNSIGNED_INT_VECTOR *>(pme_kxyz);
  Reset_List<<<3 * atom_numbers / 32 + 1, 32, 0, stream>>>(3 * atom_numbers, reinterpret_cast<int *>(PME_uxyz),
                                                           1 << 30);

  VECTOR *PME_frxyz = reinterpret_cast<VECTOR *>(pme_frxyz);
  VECTOR *frc = reinterpret_cast<VECTOR *>(force);

  std::vector<float> h_box_length(3);
  cudaMemcpyAsync(h_box_length.data(), box_length_f, sizeof(float) * h_box_length.size(), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);
  VECTOR *box_length = const_cast<VECTOR *>(reinterpret_cast<const VECTOR *>(h_box_length.data()));
  cufftComplex *PME_FQ = reinterpret_cast<cufftComplex *>(pme_fq);

  VECTOR PME_inverse_box_vector;
  PME_inverse_box_vector.x = static_cast<float>(fftx) / box_length[0].x;
  PME_inverse_box_vector.y = static_cast<float>(ffty) / box_length[0].y;
  PME_inverse_box_vector.z = static_cast<float>(fftz) / box_length[0].z;
  cufftHandle PME_plan_r2c;
  cufftHandle PME_plan_c2r;
  cufftPlan3d(&PME_plan_r2c, fftx, ffty, fftz, CUFFT_R2C);
  cufftPlan3d(&PME_plan_c2r, fftx, ffty, fftz, CUFFT_C2R);
  cufftSetStream(PME_plan_r2c, stream);
  cufftSetStream(PME_plan_c2r, stream);
  thread_PME.x = 8;
  thread_PME.y = 8;
  int PME_Nin = ffty * fftz;
  int PME_Nfft = fftx * ffty * (fftz / 2 + 1);
  int PME_Nall = fftx * ffty * fftz;
  float volume = box_length[0].x * box_length[0].y * box_length[0].z;

  UNSIGNED_INT_VECTOR *PME_kxyz_cpu;
  Malloc_Safely(reinterpret_cast<void **>(&PME_kxyz_cpu), sizeof(UNSIGNED_INT_VECTOR) * 64);

  int kx, ky, kz, kxrp, kyrp, kzrp, index;
  for (kx = 0; kx < 4; kx++) {
    for (ky = 0; ky < 4; ky++) {
      for (kz = 0; kz < 4; kz++) {
        index = kx * 16 + ky * 4 + kz;
        PME_kxyz_cpu[index].uint_x = kx;
        PME_kxyz_cpu[index].uint_y = ky;
        PME_kxyz_cpu[index].uint_z = kz;
      }
    }
  }
  cudaMemcpyAsync(PME_kxyz, PME_kxyz_cpu, sizeof(UNSIGNED_INT_VECTOR) * 64, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
  free(PME_kxyz_cpu);

  // initial start
  float *B1, *B2, *B3, *PME_BC0;
  B1 = reinterpret_cast<float *>(malloc(sizeof(float) * fftx));
  B2 = reinterpret_cast<float *>(malloc(sizeof(float) * ffty));
  B3 = reinterpret_cast<float *>(malloc(sizeof(float) * fftz));
  PME_BC0 = reinterpret_cast<float *>(malloc(sizeof(float) * PME_Nfft));

  for (kx = 0; kx < fftx; kx++) {
    B1[kx] = getb(kx, fftx, 4);
  }

  for (ky = 0; ky < ffty; ky++) {
    B2[ky] = getb(ky, ffty, 4);
  }

  for (kz = 0; kz < fftz; kz++) {
    B3[kz] = getb(kz, fftz, 4);
  }
  float mprefactor = PI * PI / -beta / beta;
  float msq;
  for (kx = 0; kx < fftx; kx++) {
    kxrp = kx;
    if (kx > fftx / 2) kxrp = fftx - kx;
    for (ky = 0; ky < ffty; ky++) {
      kyrp = ky;
      if (ky > ffty / 2) kyrp = ffty - ky;
      for (kz = 0; kz <= fftz / 2; kz++) {
        kzrp = kz;

        msq = kxrp * kxrp / box_length[0].x / box_length[0].x + kyrp * kyrp / box_length[0].y / box_length[0].y +
              kzrp * kzrp / box_length[0].z / box_length[0].z;
        index = kx * ffty * (fftz / 2 + 1) + ky * (fftz / 2 + 1) + kz;
        if ((kx + ky + kz) == 0) {
          PME_BC0[index] = 0;
        } else {
          PME_BC0[index] = 1.0 / PI / msq * exp(mprefactor * msq) / volume;
        }

        PME_BC0[index] *= B1[kx] * B2[ky] * B3[kz];
      }
    }
  }

  cudaMemcpyAsync(PME_BC, PME_BC0, sizeof(float) * PME_Nfft, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
  free(B1);
  free(B2);
  free(B3);
  free(PME_BC0);

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

void PMEReciprocalForce(int fftx, int ffty, int fftz, int atom_numbers, float beta, float *PME_BC, int *pme_uxyz,
                        float *pme_frxyz, float *PME_Q, float *pme_fq, int *PME_atom_near, int *pme_kxyz,
                        const float *box_length_f, const int *uint_crd_f, const float *charge, float *force,
                        cudaStream_t stream);

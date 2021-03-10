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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_PME_PME_COMMON_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_PME_PME_COMMON_H_
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
__constant__ float PME_Ma[4] = {1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0};
__constant__ float PME_Mb[4] = {0, 0.5, -1, 0.5};
__constant__ float PME_Mc[4] = {0, 0.5, 0, -0.5};
__constant__ float PME_Md[4] = {0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0};
__constant__ float PME_dMa[4] = {0.5, -1.5, 1.5, -0.5};
__constant__ float PME_dMb[4] = {0, 1, -2, 1};
__constant__ float PME_dMc[4] = {0, 0.5, 0, -0.5};
#define PI 3.1415926
const float periodic_factor_inverse = 2.32830643e-10;
static dim3 thread_PME;

const float cutoff = 10.0;
const float tolerance = 0.00001;

static float M_(float u, int n) {
  if (n == 2) {
    if (u > 2 || u < 0) return 0;
    return 1 - abs(u - 1);
  } else {
    return u / (n - 1) * M_(u, n - 1) + (n - u) / (n - 1) * M_(u - 1, n - 1);
  }
}

static float Get_Beta(float cutoff, float tolerance) {
  float beta, low, high, tempf;
  int ilow, ihigh;

  high = 1.0;
  ihigh = 1;

  while (1) {
    tempf = erfc(high * cutoff) / cutoff;
    if (tempf <= tolerance) break;
    high *= 2;
    ihigh++;
  }

  ihigh += 50;
  low = 0.0;
  for (ilow = 1; ilow < ihigh; ilow++) {
    beta = (low + high) / 2;
    tempf = erfc(beta * cutoff) / cutoff;
    if (tempf >= tolerance)
      low = beta;
    else
      high = beta;
  }
  return beta;
}

static cufftComplex expc(cufftComplex z) {
  cufftComplex res;
  float t = expf(z.x);
  sincosf(z.y, &res.y, &res.x);
  res.x *= t;
  res.y *= t;
  return res;
}

static float getb(int k, int NFFT, int B_order) {
  cufftComplex tempc, tempc2, res;
  float tempf;
  tempc2.x = 0;
  tempc2.y = 0;

  tempc.x = 0;
  tempc.y = 2 * (B_order - 1) * PI * k / NFFT;
  res = expc(tempc);

  for (int kk = 0; kk < (B_order - 1); kk++) {
    tempc.x = 0;
    tempc.y = 2 * PI * k / NFFT * kk;
    tempc = expc(tempc);
    tempf = M_(kk + 1, B_order);
    tempc2.x += tempf * tempc.x;
    tempc2.y += tempf * tempc.y;
  }
  res = cuCdivf(res, tempc2);
  return res.x * res.x + res.y * res.y;
}

__global__ static void PME_Atom_Near(const UNSIGNED_INT_VECTOR *uint_crd, int *PME_atom_near, const int PME_Nin,
                                     const float periodic_factor_inverse_x, const float periodic_factor_inverse_y,
                                     const float periodic_factor_inverse_z, const int atom_numbers, const int fftx,
                                     const int ffty, const int fftz, const UNSIGNED_INT_VECTOR *PME_kxyz,
                                     UNSIGNED_INT_VECTOR *PME_uxyz, VECTOR *PME_frxyz) {
  int atom = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom < atom_numbers) {
    UNSIGNED_INT_VECTOR *temp_uxyz = &PME_uxyz[atom];
    int k, tempux, tempuy, tempuz;
    float tempf;
    tempf = static_cast<float> (uint_crd[atom].uint_x) * periodic_factor_inverse_x;
    tempux = static_cast<int> (tempf);
    PME_frxyz[atom].x = tempf - tempux;

    tempf = static_cast<float> (uint_crd[atom].uint_y) * periodic_factor_inverse_y;
    tempuy = static_cast<int> (tempf);
    PME_frxyz[atom].y = tempf - tempuy;

    tempf = static_cast<float> (uint_crd[atom].uint_z) * periodic_factor_inverse_z;
    tempuz = static_cast<int> (tempf);
    PME_frxyz[atom].z = tempf - tempuz;

    if (tempux != (*temp_uxyz).uint_x || tempuy != (*temp_uxyz).uint_y || tempuz != (*temp_uxyz).uint_z) {
      (*temp_uxyz).uint_x = tempux;
      (*temp_uxyz).uint_y = tempuy;
      (*temp_uxyz).uint_z = tempuz;
      int *temp_near = PME_atom_near + atom * 64;
      int kx, ky, kz;
      for (k = 0; k < 64; k++) {
        UNSIGNED_INT_VECTOR temp_kxyz = PME_kxyz[k];
        kx = tempux - temp_kxyz.uint_x;
        if (kx < 0) kx += fftx;
        ky = tempuy - temp_kxyz.uint_y;
        if (ky < 0) ky += ffty;
        kz = tempuz - temp_kxyz.uint_z;
        if (kz < 0) kz += fftz;
        temp_near[k] = kx * PME_Nin + ky * fftz + kz;
      }
    }
  }
}

__global__ static void PME_Q_Spread(int *PME_atom_near, const float *charge, const VECTOR *PME_frxyz, float *PME_Q,
                                    const UNSIGNED_INT_VECTOR *PME_kxyz, const int atom_numbers) {
  int atom = blockDim.x * blockIdx.x + threadIdx.x;

  if (atom < atom_numbers) {
    int k;
    float tempf, tempQ, tempf2;

    int *temp_near = PME_atom_near + atom * 64;
    VECTOR temp_frxyz = PME_frxyz[atom];
    float tempcharge = charge[atom];

    UNSIGNED_INT_VECTOR temp_kxyz;
    unsigned int kx;

    for (k = threadIdx.y; k < 64; k = k + blockDim.y) {
      temp_kxyz = PME_kxyz[k];
      kx = temp_kxyz.uint_x;
      tempf = (temp_frxyz.x);
      tempf2 = tempf * tempf;
      tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];

      tempQ = tempcharge * tempf;

      kx = temp_kxyz.uint_y;
      tempf = (temp_frxyz.y);
      tempf2 = tempf * tempf;
      tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];

      tempQ = tempQ * tempf;

      kx = temp_kxyz.uint_z;
      tempf = (temp_frxyz.z);
      tempf2 = tempf * tempf;
      tempf = PME_Ma[kx] * tempf * tempf2 + PME_Mb[kx] * tempf2 + PME_Mc[kx] * tempf + PME_Md[kx];
      tempQ = tempQ * tempf;

      atomicAdd(&PME_Q[temp_near[k]], tempQ);
    }
  }
}

__global__ static void PME_Direct_Energy(const int atom_numbers, const NEIGHBOR_LIST *nl,
                                         const UNSIGNED_INT_VECTOR *uint_crd, const VECTOR *boxlength,
                                         const float *charge, const float beta, const float cutoff_square,
                                         float *direct_ene) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    NEIGHBOR_LIST nl_i = nl[atom_i];
    int N = nl_i.atom_numbers;
    int atom_j;
    int int_x;
    int int_y;
    int int_z;
    UNSIGNED_INT_VECTOR r1 = uint_crd[atom_i], r2;
    VECTOR dr;
    float dr2;
    float dr_abs;
    // float dr_inverse;
    float ene_temp;
    float charge_i = charge[atom_i];
    float ene_lin = 0.;

    // int x, y;
    // int atom_pair_LJ_type;
    for (int j = threadIdx.y; j < N; j = j + blockDim.y) {
      atom_j = nl_i.atom_serial[j];
      r2 = uint_crd[atom_j];

      int_x = r2.uint_x - r1.uint_x;
      int_y = r2.uint_y - r1.uint_y;
      int_z = r2.uint_z - r1.uint_z;
      dr.x = boxlength[0].x * int_x;
      dr.y = boxlength[0].y * int_y;
      dr.z = boxlength[0].z * int_z;

      dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
      if (dr2 < cutoff_square) {
        dr_abs = norm3df(dr.x, dr.y, dr.z);
        ene_temp = charge_i * charge[atom_j] * erfcf(beta * dr_abs) / dr_abs;
        ene_lin = ene_lin + ene_temp;
      }
    }
    atomicAdd(direct_ene, ene_lin);
  }
}

#endif

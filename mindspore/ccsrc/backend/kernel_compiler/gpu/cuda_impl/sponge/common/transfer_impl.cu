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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common/transfer_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

__device__ __host__ float fc(float Rij) {
  const float PI = 3.141592654;
  const float Rc = 1000.0;
  return 0.5 * cosf(PI / Rc * Rij) + 0.5;
}

__global__ void G_Radial(const int start_serial, const int end_serial, const float *crd, float *g_radial) {
  const float Rs = 0.5, Eta = 0.5;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= start_serial && i < end_serial) {
    float rij;
    float g_radial_lin = 0.;
    for (int j = start_serial; j < end_serial; j = j + 1) {
      if (j != i) {
        // rij = sqrtf((crd[3*i+0] - crd[j]) * (crd[i] - crd[j]));
        rij = sqrtf(normfloat(crd, crd, i, j));
        g_radial_lin = g_radial_lin + expf(-Eta * (rij - Rs) * (rij - Rs)) * fc(rij);
      } else {
        continue;
      }
    }
    g_radial[i] = g_radial_lin;
  }
}

__global__ void G_Angular(const int start_serial, const int end_serial, const float *crd, float *g_angular) {
  const float Rs = 0.5, Thetas = 3.14, Eta = 0.5, Zeta = 2.0;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= start_serial && i < end_serial) {
    float rij, rik, rjk, theta_jik;
    float g_angular_lin = 0.;
    for (int j = start_serial; j < end_serial; j = j + 1) {
      if (j != i) {
        rij = sqrtf(normfloat(crd, crd, i, j));
        for (int k = j + 1; k < end_serial; k = k + 1) {
          if (k != i) {
            rik = sqrtf(normfloat(crd, crd, i, k));
            rjk = sqrtf(normfloat(crd, crd, j, k));
            theta_jik =
              acosf(fmaxf(fminf((rij * rij + rik * rik - rjk * rjk) / (2. * rij * rik), 0.999999), -0.999999));
            g_angular_lin = g_angular_lin + powf(1. + cosf(theta_jik - Thetas), Zeta) *
                                              expf(-Eta * powf(0.5 * (rij + rik) - Rs, 2.)) * fc(rij) * fc(rik);
          } else {
            continue;
          }
        }
      } else {
        continue;
      }
    }
    g_angular[i] = powf(2., 1. - Zeta) * g_angular_lin;
  }
}

void Transfer(int start_serial, int end_serial, int number, const float *crd_f, float *g_radial, float *g_angular,
              cudaStream_t stream) {
  G_Radial<<<1, number, 0, stream>>>(start_serial, end_serial, crd_f, g_radial);
  G_Angular<<<1, number, 0, stream>>>(start_serial, end_serial, crd_f, g_angular);
  return;
}

void Transfer(int start_serial, int end_serial, int number, const float *crd_f, float *g_radial, float *g_angular,
              cudaStream_t stream);

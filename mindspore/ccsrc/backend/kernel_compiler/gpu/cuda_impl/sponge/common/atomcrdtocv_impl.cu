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

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common/atomcrdtocv_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common_sponge.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

__device__ __host__ float fc(float Rij) {
  const float PI = 3.141592654;
  const float Rc = 1000.0;
  return 0.5 * cosf(PI / Rc * Rij) + 0.5;
}

__global__ void Record_Box_Map_Times(int atom_numbers, const float *crd, const float *old_crd, float *box,
                                     int *box_map_times) {
  float half_box[3] = {0.5F * box[0], 0.5F * box[1], 0.5F * box[2]};
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < atom_numbers) {
    if (crd[3 * i + 0] - old_crd[3 * i + 0] > half_box[0]) {
      box_map_times[3 * i + 0] = box_map_times[3 * i + 0] - 1;
    } else if (crd[3 * i + 0] - old_crd[3 * i + 0] < -half_box[0]) {
      box_map_times[3 * i + 0] = box_map_times[3 * i + 0] + 1;
    }
    if (crd[3 * i + 1] - old_crd[3 * i + 1] > half_box[1]) {
      box_map_times[3 * i + 1] = box_map_times[3 * i + 1] - 1;
    } else if (crd[3 * i + 1] - old_crd[3 * i + 1] < -half_box[1]) {
      box_map_times[3 * i + 1] = box_map_times[3 * i + 1] + 1;
    }
    if (crd[3 * i + 2] - old_crd[3 * i + 2] > half_box[2]) {
      box_map_times[3 * i + 2] = box_map_times[3 * i + 2] - 1;
    } else if (crd[3 * i + 2] - old_crd[3 * i + 2] < -half_box[2]) {
      box_map_times[3 * i + 2] = box_map_times[3 * i + 2] + 1;
    }
  }
}

__global__ void gen_nowarp_crd(int atom_numbers, const float *crd, float *box, int *box_map_times, float *nowarp_crd) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < atom_numbers) {
    nowarp_crd[3 * i + 0] = static_cast<float>(box_map_times[3 * i + 0]) * box[0] + crd[3 * i + 0];
    nowarp_crd[3 * i + 1] = static_cast<float>(box_map_times[3 * i + 1]) * box[1] + crd[3 * i + 1];
    nowarp_crd[3 * i + 2] = static_cast<float>(box_map_times[3 * i + 2]) * box[2] + crd[3 * i + 2];
  }
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

void AtomCrdToCV(int atom_numbers, int start_serial, int end_serial, int number, const float *crd_f,
                 const float *old_crd, float *nowarp_crd, int *box_map_times, float *box, float *g_radial,
                 float *g_angular, cudaStream_t stream) {
  Reset_List<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(3 * atom_numbers, box_map_times,
                                                                                     0);
  Record_Box_Map_Times<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(
    atom_numbers, crd_f, old_crd, box, box_map_times);
  gen_nowarp_crd<<<ceilf(static_cast<float>(3. * atom_numbers) / 128), 128, 0, stream>>>(atom_numbers, crd_f, box,
                                                                                         box_map_times, nowarp_crd);
  G_Radial<<<1, number, 0, stream>>>(start_serial, end_serial, nowarp_crd, g_radial);
  G_Angular<<<1, number, 0, stream>>>(start_serial, end_serial, nowarp_crd, g_angular);
  return;
}

void AtomCrdToCV(int atom_numbers, int start_serial, int end_serial, int number, const float *crd_f,
                 const float *old_crd, float *nowarp_crd, int *box_map_times, float *box, float *g_radial,
                 float *g_angular, cudaStream_t stream);

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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_SPONGE_COMMONHW_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_SPONGE_COMMONHW_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "runtime/device/gpu/cuda_common.h"

#define CONSTANT_Pi 3.1415926535897932

struct VECTOR {
  float x;
  float y;
  float z;
};
struct UNSIGNED_INT_VECTOR {
  unsigned int uint_x;
  unsigned int uint_y;
  unsigned int uint_z;
};
__device__ __host__ static inline VECTOR Get_Periodic_Displacement(const UNSIGNED_INT_VECTOR uvec_a,
                                                                   const UNSIGNED_INT_VECTOR uvec_b,
                                                                   const VECTOR scaler) {
  VECTOR dr;
  dr.x = (static_cast<int>(uvec_a.uint_x - uvec_b.uint_x)) * scaler.x;
  dr.y = (static_cast<int>(uvec_a.uint_y - uvec_b.uint_y)) * scaler.y;
  dr.z = (static_cast<int>(uvec_a.uint_z - uvec_b.uint_z)) * scaler.z;
  return dr;
}

__device__ __host__ static inline VECTOR operator+(const VECTOR &veca, const VECTOR &vecb) {
  VECTOR vec;
  vec.x = veca.x + vecb.x;
  vec.y = veca.y + vecb.y;
  vec.z = veca.z + vecb.z;
  return vec;
}
__device__ __host__ static inline float operator*(const VECTOR &veca, const VECTOR &vecb) {
  return veca.x * vecb.x + veca.y * vecb.y + veca.z * vecb.z;
}
__device__ __host__ static inline VECTOR operator*(const float &a, const VECTOR &vecb) {
  VECTOR vec;
  vec.x = a * vecb.x;
  vec.y = a * vecb.y;
  vec.z = a * vecb.z;
  return vec;
}

__device__ __host__ static inline VECTOR operator-(const VECTOR &veca, const VECTOR &vecb) {
  VECTOR vec;
  vec.x = veca.x - vecb.x;
  vec.y = veca.y - vecb.y;
  vec.z = veca.z - vecb.z;
  return vec;
}

__device__ __host__ static inline VECTOR operator-(const VECTOR &vecb) {
  VECTOR vec;
  vec.x = -vecb.x;
  vec.y = -vecb.y;
  vec.z = -vecb.z;
  return vec;
}

__device__ __host__ static inline VECTOR operator^(const VECTOR &veca, const VECTOR &vecb) {
  VECTOR vec;
  vec.x = veca.y * vecb.z - veca.z * vecb.y;
  vec.y = veca.z * vecb.x - veca.x * vecb.z;
  vec.z = veca.x * vecb.y - veca.y * vecb.x;
  return vec;
}

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMP_SPONGE_COMMON_H_

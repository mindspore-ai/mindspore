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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_COMMON_SPONGE_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_COMMON_SPONGE_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <curand_kernel.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "runtime/device/gpu/cuda_common.h"

#define TWO_DIVIDED_BY_SQRT_PI 1.1283791670218446
#define CONSTANT_kB 0.00198716
#define CONSTANT_Pi 3.1415926535897932f
static dim3 thread_LJ(8, 32);

__constant__ float XRD3D_Ma[4] = {1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0};
__constant__ float XRD3D_Mb[4] = {0, 0.5, -1, 0.5};
__constant__ float XRD3D_Mc[4] = {0, 0.5, 0, -0.5};
__constant__ float XRD3D_Md[4] = {0, 1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0};
__constant__ float XRD3D_dMa[4] = {0.5, -1.5, 1.5, -0.5};
__constant__ float XRD3D_dMb[4] = {0, 1, -2, 1};
__constant__ float XRD3D_dMc[4] = {0, 0.5, 0, -0.5};

struct VECTOR {
  float x;
  float y;
  float z;
};

struct INT_VECTOR {
  int int_x;
  int int_y;
  int int_z;
};

struct UNSIGNED_INT_VECTOR {
  unsigned int uint_x;
  unsigned int uint_y;
  unsigned int uint_z;
};

struct NEIGHBOR_LIST {
  int atom_numbers;
  int *atom_serial;
};
struct UINT_VECTOR_LJ_TYPE {
  unsigned int uint_x;
  unsigned int uint_y;
  unsigned int uint_z;
  int LJ_type;
  float charge;
};

struct ATOM_NEAR {
  int *atom_serial;
};

struct GRID_BUCKET {
  int *atom_serial;
};

struct GRID_POINTER {
  int *grid_serial;
};

struct VIRTUAL_TYPE_0 {
  float virtual_atom;
  float from_1;
  float h_double;
};

struct VIRTUAL_TYPE_1 {
  float virtual_atom;
  float from_1;
  float from_2;
  float a;
};

struct VIRTUAL_TYPE_2 {
  float virtual_atom;
  float from_1;
  float from_2;
  float from_3;
  float a;
  float b;
};

struct VIRTUAL_TYPE_3 {
  float virtual_atom;
  float from_1;
  float from_2;
  float from_3;
  float d;
  float k;
};

struct CONSTRAIN_PAIR {
  int atom_i_serial;
  int atom_j_serial;
  float constant_r;
  float constrain_k;
};

__device__ __host__ static inline VECTOR operator-(const VECTOR &veca, const VECTOR &vecb) {
  VECTOR vec;
  vec.x = veca.x - vecb.x;
  vec.y = veca.y - vecb.y;
  vec.z = veca.z - vecb.z;
  return vec;
}

__device__ __host__ static inline VECTOR Get_Periodic_Displacement(const UNSIGNED_INT_VECTOR uvec_a,
                                                                   const UNSIGNED_INT_VECTOR uvec_b,
                                                                   const VECTOR scaler) {
  VECTOR dr;
  dr.x = (static_cast<int>(uvec_a.uint_x - uvec_b.uint_x)) * scaler.x;
  dr.y = (static_cast<int>(uvec_a.uint_y - uvec_b.uint_y)) * scaler.y;
  dr.z = (static_cast<int>(uvec_a.uint_z - uvec_b.uint_z)) * scaler.z;
  return dr;
}

__device__ __host__ static inline VECTOR Get_Periodic_Displacement(const UINT_VECTOR_LJ_TYPE uvec_a,
                                                                   const UINT_VECTOR_LJ_TYPE uvec_b,
                                                                   const VECTOR scaler) {
  VECTOR dr;
  dr.x = (static_cast<int>(uvec_a.uint_x - uvec_b.uint_x)) * scaler.x;
  dr.y = (static_cast<int>(uvec_a.uint_y - uvec_b.uint_y)) * scaler.y;
  dr.z = (static_cast<int>(uvec_a.uint_z - uvec_b.uint_z)) * scaler.z;
  return dr;
}

__device__ __host__ static inline VECTOR Get_Periodic_Displacement(const VECTOR vec_a, const VECTOR vec_b,
                                                                   const VECTOR box_length) {
  VECTOR dr;
  dr = vec_a - vec_b;
  dr.x = dr.x - floorf(dr.x / box_length.x + 0.5) * box_length.x;
  dr.y = dr.y - floorf(dr.y / box_length.y + 0.5) * box_length.y;
  dr.z = dr.z - floorf(dr.z / box_length.z + 0.5) * box_length.z;
  return dr;
}

__device__ __host__ static inline VECTOR Get_Periodic_Displacement(const VECTOR vec_a, const VECTOR vec_b,
                                                                   const VECTOR box_length,
                                                                   const VECTOR box_length_inverse) {
  VECTOR dr;
  dr = vec_a - vec_b;
  dr.x = dr.x - floorf(dr.x * box_length_inverse.x + 0.5) * box_length.x;
  dr.y = dr.y - floorf(dr.y * box_length_inverse.y + 0.5) * box_length.y;
  dr.z = dr.z - floorf(dr.z * box_length_inverse.z + 0.5) * box_length.z;
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

__device__ __host__ static inline float normfloat(const float *x, const float *y, int i, int j) {
  float s = 0;
  s += (x[3 * i + 0] - y[3 * j + 0]) * (x[3 * i + 0] - y[3 * j + 0]);
  s += (x[3 * i + 1] - y[3 * j + 1]) * (x[3 * i + 1] - y[3 * j + 1]);
  s += (x[3 * i + 2] - y[3 * j + 2]) * (x[3 * i + 2] - y[3 * j + 2]);
  return s;
}

__global__ static void construct_neighbor_list_kernel(int atom_numbers, int max_neighbor_numbers, int *nl_atom_numbers,
                                                      int *nl_atom_serial, NEIGHBOR_LIST *nl) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i += gridDim.x * blockDim.x) {
    nl[i].atom_numbers = nl_atom_numbers[i];
    nl[i].atom_serial = nl_atom_serial + i * max_neighbor_numbers;
  }
}

__global__ static void construct_atom_near(int atom_numbers, int near_numbers, int *atom_serial, ATOM_NEAR *an) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < atom_numbers; i += gridDim.x * blockDim.x) {
    an[i].atom_serial = atom_serial + i * near_numbers;
  }
}

static inline bool Malloc_Safely(void **address, size_t size) {
  address[0] = NULL;
  address[0] = reinterpret_cast<void *>(malloc(size));
  if (address[0] != NULL) {
    return true;
  } else {
    printf("malloc failed!\n");
    getchar();
    return false;
  }
}

static inline bool Cuda_Malloc_Safely(void **address, size_t size) {
  cudaError_t cuda_error = cudaMalloc(&address[0], size);
  if (cuda_error == 0) {
    return true;
  } else {
    printf("cudaMalloc failed! error %d\n", cuda_error);
    getchar();
    return false;
  }
}

__global__ static void construct_constrain_pair(int constrain_pair_numbers, const int *atom_i_serials,
                                                const int *atom_j_serials, const float *constant_rs,
                                                const float *constrain_ks, CONSTRAIN_PAIR *constrain_pair) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < constrain_pair_numbers) {
    constrain_pair[atom_i].atom_i_serial = atom_i_serials[atom_i];
    constrain_pair[atom_i].atom_j_serial = atom_j_serials[atom_i];
    constrain_pair[atom_i].constant_r = constant_rs[atom_i];
    constrain_pair[atom_i].constrain_k = constrain_ks[atom_i];
  }
}

__global__ static void Copy_Crd_To_New_Crd_Start(const int atom_numbers, const UNSIGNED_INT_VECTOR *crd,
                                                 UINT_VECTOR_LJ_TYPE *new_crd, const int *LJ_type,
                                                 const float *charge) {
  int atom_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (atom_i < atom_numbers) {
    new_crd[atom_i].uint_x = crd[atom_i].uint_x;
    new_crd[atom_i].uint_y = crd[atom_i].uint_y;
    new_crd[atom_i].uint_z = crd[atom_i].uint_z;
    new_crd[atom_i].LJ_type = LJ_type[atom_i];
    new_crd[atom_i].charge = charge[atom_i];
  }
}

// void Constrain_Force_Cycle_With_Virial(int atom_numbers, int constrain_pair_numbers, const unsigned int *uint_crd_f,
//                                        const float *scaler_f, float *constrain_pair_f, const float *pair_dr_f,
//                                        const int *atom_i_serials, const int *atom_j_serials, const float
//                                        *constant_rs, const float *constrain_ks, float *test_frc_f, float
//                                        *d_atom_virial, cudaStream_t stream);

__global__ static void Rand_Normal(const int float4_numbers, curandStatePhilox4_32_10_t *rand_state,
                                   float4 *rand_float4) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < float4_numbers) {
    rand_float4[i] = curand_normal4(&rand_state[i]);
  }
}

__global__ static void Setup_Rand_Normal_Kernel(const int float4_numbers, curandStatePhilox4_32_10_t *rand_state,
                                                const int seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets same seed, a different sequence
  number, no offset */
  if (id < float4_numbers) {
    curand_init(seed, id, 0, &rand_state[id]);
  }
}

__global__ static void Reset_List(const int element_numbers, int *list, const int replace_element) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = replace_element;
  }
}

__global__ static void Reset_List(const int element_numbers, float *list, const float replace_element) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = replace_element;
  }
}

__global__ static void Sum_Of_List(const int element_numbers, const float *list, float *sum) {
  if (threadIdx.x == 0) {
    sum[0] = 0.;
  }
  __syncthreads();
  float lin = 0.;
  for (int i = threadIdx.x; i < element_numbers; i = i + blockDim.x) {
    lin = lin + list[i];
  }
  atomicAdd(sum, lin);
}

__global__ static void Scale_List(const int element_numbers, float *list, float scaler) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = list[i] * scaler;
  }
}

__global__ static void Copy_List(const int element_numbers, const int *origin_list, int *list) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = origin_list[i];
  }
}
__global__ static void Copy_List(const int element_numbers, const float *origin_list, float *list) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < element_numbers) {
    list[i] = origin_list[i];
  }
}

__global__ static void Print(const size_t size, const float *input_x) {
  for (size_t i = 0; i < size; i++) {
    printf("%f\n", input_x[i]);
  }
  return;
}
__global__ static void Print(const size_t size, const int *input_x) {
  for (size_t i = 0; i < size; i++) {
    printf("%d\n", input_x[i]);
  }
  return;
}

__device__ static VECTOR Make_Vector_Not_Exceed_Value(VECTOR vector, const float value) {
  return fminf(1.0, value * rnorm3df(vector.x, vector.y, vector.z)) * vector;
}

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_COMMON_SPONGE_H_

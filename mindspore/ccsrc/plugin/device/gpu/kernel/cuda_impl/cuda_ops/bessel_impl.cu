/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <math.h>
#include <cuda_runtime.h>
#include "bessel_impl.cuh"
#include "include/cuda_fp16.h"

__constant__ float MAXNUM = INFINITY;

__constant__ float NUM_TWO = 2.0;

__constant__ float BESSELK0_A_F[7] = {
  1.90451637722020886025E-9f, 2.53479107902614945675E-7f, 2.28621210311945178607E-5f, 1.26461541144692592338E-3f,
  3.59799365153615016266E-2f, 3.44289899924628486886E-1f, -5.35327393233902768720E-1f};

__constant__ float BESSELK0_B_F[10] = {
  -1.69753450938905987466E-9f, 8.57403401741422608519E-9f, -4.66048989768794782956E-8f, 2.76681363944501510342E-7f,
  -1.83175552271911948767E-6f, 1.39498137188764993662E-5f, -1.28495495816278026384E-4f, 1.56988388573005337491E-3f,
  -3.14481013119645005427E-2f, 2.44030308206595545468E0f};

__constant__ float BESSELK1_A_F[7] = {
  -2.21338763073472585583E-8f, -2.43340614156596823496E-6f, -1.73028895751305206302E-4f, -6.97572385963986435018E-3f,
  -1.22611180822657148235E-1f, -3.53155960776544875667E-1f, 1.52530022733894777053E0f};

__constant__ float BESSELK1_B_F[10] = {
  2.01504975519703286596E-9f, -1.03457624656780970260E-8f, 5.74108412545004946722E-8f, -3.50196060308781257119E-7f,
  2.40648494783721712015E-6f, -1.93619797416608296024E-5f, 1.95215518471351631108E-4f, -2.85781685962277938680E-3f,
  1.03923736576817238437E-1f, 2.72062619048444266945E0f};

__constant__ double BESSELK0_A_D[10] = {
  1.37446543561352307156E-16, 4.25981614279661018399E-14, 1.03496952576338420167E-11, 1.90451637722020886025E-9,
  2.53479107902614945675E-7,  2.28621210311945178607E-5,  1.26461541144692592338E-3,  3.59799365153615016266E-2,
  3.44289899924628486886E-1,  -5.35327393233902768720E-1};

__constant__ double BESSELK0_B_D[25] = {
  5.30043377268626276149E-18, -1.64758043015242134646E-17, 5.21039150503902756861E-17, -1.67823109680541210385E-16,
  5.51205597852431940784E-16, -1.84859337734377901440E-15, 6.34007647740507060557E-15, -2.22751332699166985548E-14,
  8.03289077536357521100E-14, -2.98009692317273043925E-13, 1.14034058820847496303E-12, -4.51459788337394416547E-12,
  1.85594911495471785253E-11, -7.95748924447710747776E-11, 3.57739728140030116597E-10, -1.69753450938905987466E-9,
  8.57403401741422608519E-9,  -4.66048989768794782956E-8,  2.76681363944501510342E-7,  -1.83175552271911948767E-6,
  1.39498137188764993662E-5,  -1.28495495816278026384E-4,  1.56988388573005337491E-3,  -3.14481013119645005427E-2,
  2.44030308206595545468E0};

__constant__ double BESSELK1_A_D[11] = {
  -7.02386347938628759343E-18, -2.42744985051936593393E-15, -6.66690169419932900609E-13, -1.41148839263352776110E-10,
  -2.21338763073472585583E-8,  -2.43340614156596823496E-6,  -1.73028895751305206302E-4,  -6.97572385963986435018E-3,
  -1.22611180822657148235E-1,  -3.53155960776544875667E-1,  1.52530022733894777053E0};

__constant__ double BESSELK1_B_D[25] = {
  -5.75674448366501715755E-18, 1.79405087314755922667E-17, -5.68946255844285935196E-17, 1.83809354436663880070E-16,
  -6.05704724837331885336E-16, 2.03870316562433424052E-15, -7.01983709041831346144E-15, 2.47715442448130437068E-14,
  -8.97670518232499435011E-14, 3.34841966607842919884E-13, -1.28917396095102890680E-12, 5.13963967348173025100E-12,
  -2.12996783842756842877E-11, 9.21831518760500529508E-11, -4.19035475934189648750E-10, 2.01504975519703286596E-9,
  -1.03457624656780970260E-8,  5.74108412545004946722E-8,  -3.50196060308781257119E-7,  2.40648494783721712015E-6,
  -1.93619797416608296024E-5,  1.95215518471351631108E-4,  -2.85781685962277938680E-3,  1.03923736576817238437E-1,
  2.72062619048444266945E0};

template <typename T, int N>
__device__ __forceinline__ T Chebevl(T x, T coef[]) {
  T b0 = coef[0];
  T b1 = static_cast<T>(0.0f);
  T b2;
  for (size_t i = 1; i < N; i++) {
    b2 = b1;
    b1 = b0;
    b0 = ((x * b1) + coef[i] - b2);
  }
  return (static_cast<T>(0.5f)) * (b0 - b2);
}

template <typename T>
__device__ __forceinline__ T bessel_k0_le_two(T x) {
  return static_cast<T>(bessel_k0_le_two(static_cast<float>(x)));
}

template <typename T>
__device__ __forceinline__ T bessel_k0_gt_two(T x) {
  return static_cast<T>(bessel_k0_gt_two(static_cast<float>(x)));
}

template <>
__device__ __forceinline__ float bessel_k0_le_two(float x) {
  float x_le_two = x * x - static_cast<float>(2.0);
  x_le_two = Chebevl<float, 7>(x_le_two, BESSELK0_A_F);
  x_le_two = ((-logf(x / 2.0)) * (cyl_bessel_i0f(x))) + x_le_two;
  return x_le_two;
}

template <>
__device__ __forceinline__ float bessel_k0_gt_two(float x) {
  float x_gt_two = static_cast<float>(8.0) / x - NUM_TWO;
  x_gt_two = Chebevl<float, 10>(x_gt_two, BESSELK0_B_F);
  x_gt_two = (x_gt_two * expf(-x)) * __frsqrt_rn(x);
  return x_gt_two;
}

template <>
__device__ __forceinline__ double bessel_k0_le_two(double x) {
  double x_le_two = x * x - static_cast<double>(2.0);
  x_le_two = Chebevl<double, 10>(x_le_two, BESSELK0_A_D);
  x_le_two = ((-log(x / 2.0)) * (cyl_bessel_i0(x))) + x_le_two;
  return x_le_two;
}

template <>
__device__ __forceinline__ double bessel_k0_gt_two(double x) {
  double x_gt_two = static_cast<double>(8.0) / x - static_cast<double>(NUM_TWO);
  x_gt_two = Chebevl<double, 25>(x_gt_two, BESSELK0_B_D);
  x_gt_two = (x_gt_two * exp(-x)) * rsqrt(x);
  return x_gt_two;
}

template <typename T>
__device__ __forceinline__ T bessel_k0e_le_two(T x) {
  return static_cast<T>(bessel_k0e_le_two(static_cast<float>(x)));
}

template <typename T>
__device__ __forceinline__ T bessel_k0e_gt_two(T x) {
  return static_cast<T>(bessel_k0e_gt_two(static_cast<float>(x)));
}

template <>
__device__ __forceinline__ float bessel_k0e_le_two(float x) {
  float x_le_two = x * x - static_cast<float>(2.0);
  x_le_two = Chebevl<float, 7>(x_le_two, BESSELK0_A_F);
  x_le_two = (-logf(x / 2.0)) * cyl_bessel_i0f(x) + x_le_two;
  x_le_two = expf(x) * x_le_two;
  return x_le_two;
}

template <>
__device__ __forceinline__ float bessel_k0e_gt_two(float x) {
  float x_gt_two = static_cast<float>(8.0) / x - NUM_TWO;
  x_gt_two = Chebevl<float, 10>(x_gt_two, BESSELK0_B_F) * __frsqrt_rn(x);
  return x_gt_two;
}

template <>
__device__ __forceinline__ double bessel_k0e_le_two(double x) {
  double x_le_two = x * x - static_cast<double>(2.0);
  x_le_two = Chebevl<double, 10>(x_le_two, BESSELK0_A_D);
  x_le_two = (-log(x / 2.0)) * cyl_bessel_i0(x) + x_le_two;
  x_le_two = exp(x) * x_le_two;
  return x_le_two;
}

template <>
__device__ __forceinline__ double bessel_k0e_gt_two(double x) {
  double x_gt_two = static_cast<double>(8.0) / x - NUM_TWO;
  x_gt_two = Chebevl<double, 25>(x_gt_two, BESSELK0_B_D) * rsqrt(x);
  return x_gt_two;
}

template <typename T>
__device__ __forceinline__ T bessel_k1_le_two(T x) {
  return static_cast<T>(bessel_k1_le_two(static_cast<float>(x)));
}

template <typename T>
__device__ __forceinline__ T bessel_k1_gt_two(T x) {
  return static_cast<T>(bessel_k1_gt_two(static_cast<float>(x)));
}

template <>
__device__ __forceinline__ float bessel_k1_le_two(float x) {
  float x_le_two = x * x - static_cast<float>(2.0);
  x_le_two = Chebevl<float, 7>(x_le_two, BESSELK1_A_F) / x;
  x_le_two = (logf(x / 2.0) * (cyl_bessel_i1f(x))) + x_le_two;
  return x_le_two;
}

template <>
__device__ __forceinline__ float bessel_k1_gt_two(float x) {
  float x_gt_two = static_cast<float>(8.0) / x - NUM_TWO;
  x_gt_two = Chebevl<float, 10>(x_gt_two, BESSELK1_B_F);
  x_gt_two = x_gt_two * __frsqrt_rn(x);
  x_gt_two = expf(-x) * x_gt_two;
  return x_gt_two;
}

template <>
__device__ __forceinline__ double bessel_k1_le_two(double x) {
  double x_le_two = x * x - static_cast<double>(2.0);
  x_le_two = Chebevl<double, 11>(x_le_two, BESSELK1_A_D) / x;
  x_le_two = (log(x / 2.0) * (cyl_bessel_i1(x))) + x_le_two;
  return x_le_two;
}

template <>
__device__ __forceinline__ double bessel_k1_gt_two(double x) {
  double x_gt_two = static_cast<double>(8.0) / x - NUM_TWO;
  x_gt_two = Chebevl<double, 25>(x_gt_two, BESSELK1_B_D);
  x_gt_two = x_gt_two * rsqrt(x);
  x_gt_two = exp(-x) * x_gt_two;
  return x_gt_two;
}

template <typename T>
__device__ __forceinline__ T bessel_k1e_le_two(T x) {
  return static_cast<T>(bessel_k1e_le_two(static_cast<float>(x)));
}

template <typename T>
__device__ __forceinline__ T bessel_k1e_gt_two(T x) {
  return static_cast<T>(bessel_k1e_gt_two(static_cast<float>(x)));
}

template <>
__device__ __forceinline__ float bessel_k1e_le_two(float x) {
  float x_le_two = x * x - static_cast<float>(2.0);
  x_le_two = Chebevl<float, 7>(x_le_two, BESSELK1_A_F) / x;
  x_le_two = (logf(x / 2.0) * (cyl_bessel_i1f(x))) + x_le_two;
  x_le_two = expf(x) * x_le_two;
  return x_le_two;
}

template <>
__device__ __forceinline__ float bessel_k1e_gt_two(float x) {
  float x_gt_two = static_cast<float>(8.0) / x - NUM_TWO;
  x_gt_two = Chebevl<float, 10>(x_gt_two, BESSELK1_B_F);
  x_gt_two = x_gt_two * __frsqrt_rn(x);
  return x_gt_two;
}

template <>
__device__ __forceinline__ double bessel_k1e_le_two(double x) {
  double x_le_two = x * x - static_cast<double>(2.0);
  x_le_two = Chebevl<double, 11>(x_le_two, BESSELK1_A_D) / x;
  x_le_two = (log(x / 2.0) * (cyl_bessel_i1(x))) + x_le_two;
  x_le_two = exp(x) * x_le_two;
  return x_le_two;
}

template <>
__device__ __forceinline__ double bessel_k1e_gt_two(double x) {
  double x_gt_two = static_cast<double>(8.0) / x - NUM_TWO;
  x_gt_two = Chebevl<double, 25>(x_gt_two, BESSELK1_B_D);
  x_gt_two = x_gt_two * rsqrt(x);
  return x_gt_two;
}

template <typename T>
__device__ __forceinline__ T bessel_i0(T x) {
  return static_cast<T>(bessel_i0(static_cast<float>(x)));
}

template <>
__device__ __forceinline__ float bessel_i0(float x) {
  return cyl_bessel_i0f(x);
}

template <>
__device__ __forceinline__ double bessel_i0(double x) {
  return cyl_bessel_i0(x);
}

template <typename T>
__device__ __forceinline__ T bessel_i0e(T x) {
  return static_cast<T>(bessel_i0e(static_cast<float>(x)));
}

template <>
__device__ __forceinline__ float bessel_i0e(float x) {
  return cyl_bessel_i0f(x) / expf(fabsf(x));
}

template <>
__device__ __forceinline__ double bessel_i0e(double x) {
  return cyl_bessel_i0(x) / exp(fabs(x));
}

template <typename T>
__device__ __forceinline__ T bessel_i1(T x) {
  return static_cast<T>(bessel_i1(static_cast<float>(x)));
}

template <>
__device__ __forceinline__ float bessel_i1(float x) {
  return cyl_bessel_i1f(x);
}

template <>
__device__ __forceinline__ double bessel_i1(double x) {
  return cyl_bessel_i1(x);
}

template <typename T>
__device__ __forceinline__ T bessel_i1e(T x) {
  return static_cast<T>(bessel_i1e(static_cast<float>(x)));
}

template <>
__device__ __forceinline__ float bessel_i1e(float x) {
  return cyl_bessel_i1f(x) / expf(fabsf(x));
}

template <>
__device__ __forceinline__ double bessel_i1e(double x) {
  return cyl_bessel_i1(x) / exp(fabs(x));
}

template <typename T>
__global__ void BesselI0(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = bessel_i0(input[pos]);
  }
  return;
}

template <typename T>
__global__ void BesselI0e(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = bessel_i0e(input[pos]);
  }
  return;
}

template <typename T>
__global__ void BesselI1(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = bessel_i1(input[pos]);
  }
  return;
}

template <typename T>
__global__ void BesselI1e(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = bessel_i1e(input[pos]);
  }
  return;
}

template <typename T>
__global__ void BesselJ0(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = j0(input[pos]);
  }
  return;
}

template <>
__global__ void BesselJ0(const size_t size, const half *input, half *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    float temp = j0(__half2float(input[pos]));
    output[pos] = __float2half(temp);
  }
  return;
}

template <typename T>
__global__ void BesselJ1(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = j1(input[pos]);
  }
  return;
}

template <>
__global__ void BesselJ1(const size_t size, const half *input, half *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    float temp = j1(__half2float(input[pos]));
    output[pos] = __float2half(temp);
  }
  return;
}

template <typename T>
__global__ void BesselY0(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = y0(input[pos]);
  }
  return;
}

template <>
__global__ void BesselY0(const size_t size, const half *input, half *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    float temp = y0(__half2float(input[pos]));
    output[pos] = __float2half(temp);
  }
  return;
}

template <typename T>
__global__ void BesselY1(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = y1(input[pos]);
  }
  return;
}

template <>
__global__ void BesselY1(const size_t size, const half *input, half *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    float temp = y1(__half2float(input[pos]));
    output[pos] = __float2half(temp);
  }
  return;
}

template <typename T>
__global__ void BesselK0(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    if (input[pos] <= static_cast<T>(NUM_TWO)) {
      if (input[pos] <= static_cast<T>(0.0)) {
        output[pos] = static_cast<T>(MAXNUM);
      } else {
        output[pos] = bessel_k0_le_two(input[pos]);
      }
    } else {
      output[pos] = bessel_k0_gt_two(input[pos]);
    }
  }
  return;
}

template <typename T>
__global__ void BesselK0e(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    if (input[pos] <= static_cast<T>(NUM_TWO)) {
      if (input[pos] <= static_cast<T>(0.0)) {
        output[pos] = static_cast<T>(MAXNUM);
      } else {
        output[pos] = bessel_k0e_le_two(input[pos]);
      }
    } else {
      output[pos] = bessel_k0e_gt_two(input[pos]);
    }
  }
  return;
}

template <typename T>
__global__ void BesselK1(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    if (input[pos] <= static_cast<T>(NUM_TWO)) {
      if (input[pos] <= static_cast<T>(0.0)) {
        output[pos] = static_cast<T>(MAXNUM);
      } else {
        output[pos] = bessel_k1_le_two(input[pos]);
      }
    } else {
      output[pos] = bessel_k1_gt_two(input[pos]);
    }
  }
  return;
}

template <typename T>
__global__ void BesselK1e(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    if (input[pos] <= static_cast<T>(NUM_TWO)) {
      if (input[pos] <= static_cast<T>(0.0)) {
        output[pos] = static_cast<T>(MAXNUM);
      } else {
        output[pos] = bessel_k1e_le_two(input[pos]);
      }
    } else {
      output[pos] = bessel_k1e_gt_two(input[pos]);
    }
  }
  return;
}

template <typename T>
cudaError_t CalBesselJ0(const size_t size, const T *input, T *output, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  BesselJ0<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <>
cudaError_t CalBesselJ0(const size_t size, const half *input, half *output, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  BesselJ0<half><<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBesselJ1(const size_t size, const T *input, T *output, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  BesselJ1<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <>
cudaError_t CalBesselJ1(const size_t size, const half *input, half *output, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  BesselJ1<half><<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBesselK0(const size_t size, const T *input, T *output, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  BesselK0<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBesselK0e(const size_t size, const T *input, T *output, const uint32_t &device_id,
                         cudaStream_t cuda_stream) {
  BesselK0e<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBesselK1(const size_t size, const T *input, T *output, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  BesselK1<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBesselK1e(const size_t size, const T *input, T *output, const uint32_t &device_id,
                         cudaStream_t cuda_stream) {
  BesselK1e<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBesselY0(const size_t size, const T *input, T *output, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  BesselY0<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBesselY1(const size_t size, const T *input, T *output, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  BesselY1<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBesselI0(const size_t size, const T *input, T *output, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  BesselI0<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBesselI0e(const size_t size, const T *input, T *output, const uint32_t &device_id,
                         cudaStream_t cuda_stream) {
  BesselI0e<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBesselI1(const size_t size, const T *input, T *output, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  BesselI1<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalBesselI1e(const size_t size, const T *input, T *output, const uint32_t &device_id,
                         cudaStream_t cuda_stream) {
  BesselI1e<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalBesselJ0<double>(const size_t size, const double *input, double *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselJ0<float>(const size_t size, const float *input, float *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselJ0<half>(const size_t size, const half *input, half *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselJ1<double>(const size_t size, const double *input, double *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselJ1<float>(const size_t size, const float *input, float *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselJ1<half>(const size_t size, const half *input, half *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselK0<double>(const size_t size, const double *input, double *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselK0<float>(const size_t size, const float *input, float *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselK0<half>(const size_t size, const half *input, half *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselK0e<double>(const size_t size, const double *input, double *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselK0e<float>(const size_t size, const float *input, float *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselK0e<half>(const size_t size, const half *input, half *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselK1<double>(const size_t size, const double *input, double *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselK1<float>(const size_t size, const float *input, float *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselK1<half>(const size_t size, const half *input, half *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselK1e<double>(const size_t size, const double *input, double *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselK1e<float>(const size_t size, const float *input, float *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselK1e<half>(const size_t size, const half *input, half *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselI0<double>(const size_t size, const double *input, double *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselI0<float>(const size_t size, const float *input, float *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselI0<half>(const size_t size, const half *input, half *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselI0e<double>(const size_t size, const double *input, double *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselI0e<float>(const size_t size, const float *input, float *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselI0e<half>(const size_t size, const half *input, half *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselI1<double>(const size_t size, const double *input, double *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselI1<float>(const size_t size, const float *input, float *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselI1<half>(const size_t size, const half *input, half *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselI1e<double>(const size_t size, const double *input, double *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselI1e<float>(const size_t size, const float *input, float *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselI1e<half>(const size_t size, const half *input, half *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselY0<double>(const size_t size, const double *input, double *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselY0<float>(const size_t size, const float *input, float *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselY0<half>(const size_t size, const half *input, half *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselY1<double>(const size_t size, const double *input, double *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselY1<float>(const size_t size, const float *input, float *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBesselY1<half>(const size_t size, const half *input, half *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);

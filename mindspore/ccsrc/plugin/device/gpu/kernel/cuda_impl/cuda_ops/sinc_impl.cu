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
#include <complex>
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sinc_impl.cuh"

constexpr uint elements_per_thread = 4;
constexpr uint threads_per_block = 256;
constexpr uint elements_per_block = elements_per_thread * threads_per_block;

template <typename T, typename S>
struct VectorizedTrait {
  static const uint VecSizeT = 4;
  static const uint VecSizeS = 4;
};

template <>
struct VectorizedTrait<half, half> {
  static const uint VecSizeT = 2;
  static const uint VecSizeS = 2;
};

template <typename T, int VecSizeT>
struct alignas(sizeof(T) * VecSizeT) AlignVecIn {
  T datain[VecSizeT];
};

template <typename S, int VecSizeS>
struct alignas(sizeof(S) * VecSizeS) AlignVecOut {
  S dataout[VecSizeS];
};

template <typename Func, typename T, typename S>
__device__ __forceinline__ void VectorizedCall(Func func, const T *input, S *output) {
  constexpr uint vec_size_t = VectorizedTrait<T, S>::VecSizeT;
  constexpr uint vec_size_s = VectorizedTrait<T, S>::VecSizeS;
  constexpr uint elements_per_loop = elements_per_thread / vec_size_t;
  using VecT = AlignVecIn<T, vec_size_t>;
  using VecS = AlignVecOut<S, vec_size_s>;

  uint tid = threadIdx.x;

  auto vec_input = reinterpret_cast<const VecT *>(input);
  auto vec_output = reinterpret_cast<VecS *>(output);

  for (uint i = 0; i < elements_per_loop; i++) {
    uint index = tid + i * threads_per_block;
    VecT cache_in = vec_input[index];
    VecS cache_out = vec_output[index];
    for (uint j = 0; j < vec_size_t; j++) {
      cache_out.dataout[j] = func(cache_in.datain[j]);
    }
    vec_output[index] = cache_out;
    }
}

template <typename Func, typename T, typename S>
__device__ __forceinline__ void NormalCall(Func func, const T *input, S *output, uint remaining) {
  uint loop = UP_DIV(remaining, elements_per_thread);

  for (uint i = threadIdx.x; i < loop; i += blockDim.x) {
    for (uint j = 0; j < elements_per_thread; j++) {
      uint index = i * elements_per_thread + j;
      if (index >= remaining) {
        return;
      }
      output[index] = func(input[index]);
    }
  }
}

template <typename Func, typename T, typename S>
__global__ void SincVectorized(Func func, const T *input, S *output, uint num_of_elements) {
  uint offset = elements_per_block * blockIdx.x;
  uint remaining = num_of_elements - offset;

  if (blockIdx.x + 1 == gridDim.x && remaining != elements_per_block) {
    NormalCall(func, input + offset, output + offset, remaining);
  } else {
    VectorizedCall(func, input + offset, output + offset);
  }
}

template <typename T, typename S>
struct SincFunctor {
  __device__ __forceinline__ S operator()(const T input) const {
    const double PI = acos(-1.0);
    const double zero = static_cast<double>(0.0);
    const double one = static_cast<double>(1.0);
    double output = zero;
    if (static_cast<double>(input) == zero) {
      output = one;
    } else {
      double temp = PI * static_cast<double>(input);
      output = sinf(temp) / temp;
    }
    return static_cast<S>(output);
  }
};

template <>
struct SincFunctor <half, half> {
  __device__ __forceinline__ half operator()(const half input) const {
    const float PI = acos(-1.0);
    const float zero = static_cast<float>(0);
    const float one = static_cast<float>(1);
    float output = zero;
    if (__half2float(input) == zero) {
      output = one;
    } else {
      float temp = PI * static_cast<float>(__half2float(input));
      output = sinf(temp) / temp;
    }
    return __float2half(static_cast<float>(output));
  }
};

template <>
struct SincFunctor <Complex<float>, Complex<float>> {
  __device__ __forceinline__ Complex<float> operator()(const Complex<float> input) const {
    const float PI = acos(-1.0);
    const float zero = static_cast<float>(0);
    float a = input.real();
    float b = input.imag();
    Complex<float> result;
    if (a == zero && b == zero) {
      result.real(1.0);
      result.imag(0.0);
    } else {
      float tmp_a = PI * a;
      float tmp_b = PI * b;
      float A = sinf(tmp_a) * coshf(tmp_b);
      float B = cosf(tmp_a) * sinhf(tmp_b);
      float T = tmp_a * tmp_a + tmp_b * tmp_b;
      float rs_real = (A * tmp_a + B * tmp_b) / T;
      float rs_imag = (B * tmp_a - A * tmp_b) / T;
      result.real(rs_real);
      result.imag(rs_imag);
    }
    return result;
}
};

template <>
struct SincFunctor <Complex<double>, Complex<double>> {
  __device__ __forceinline__ Complex<double> operator()(const Complex<double> input) const {
    const double PI = acos(-1.0);
    const double zero = static_cast<double>(0);
    double a = input.real();
    double b = input.imag();
    Complex<double> result;
    if (a == zero && b == zero) {
      result.real(1.0);
      result.imag(0.0);
    } else {
      double tmp_a = PI * a;
      double tmp_b = PI * b;
      double A = sinf(tmp_a) * coshf(tmp_b);
      double B = cosf(tmp_a) * sinhf(tmp_b);
      double T = tmp_a * tmp_a + tmp_b * tmp_b;
      double rs_real = (A * tmp_a + B * tmp_b) / T;
      double rs_imag = (B * tmp_a - A * tmp_b) / T;

      result.real(rs_real);
      result.imag(rs_imag);
    }
    return result;
}
};

template <typename T, typename S>
void CalSinc(const size_t size, const T *input, S *output, const uint32_t &device_id,
             cudaStream_t cuda_stream) {
  SincFunctor<T, S> functor{};
  auto block_x = threads_per_block;
  auto grid_x = UP_DIV(static_cast<uint>(size), elements_per_block);
  dim3 block{block_x};
  dim3 grid{grid_x};
  SincVectorized<<<grid, block, 0, cuda_stream>>>(functor, input, output, size);
}

template
CUDA_LIB_EXPORT void CalSinc<uint8_t, float>(const size_t size, const uint8_t *input, float *output,
                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<int8_t, float>(const size_t size, const int8_t *input, float *output,
                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<uint16_t, float>(const size_t size, const uint16_t *input, float *output,
                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<int16_t, float>(const size_t size, const int16_t *input, float *output,
                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<uint32_t, float>(const size_t size, const uint32_t *input, float *output,
                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<int32_t, float>(const size_t size, const int32_t *input, float *output,
                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<uint64_t, float>(const size_t size, const uint64_t *input, float *output,
                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<int64_t, float>(const size_t size, const int64_t *input, float *output,
                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<bool, float>(const size_t size, const bool *input, float *output,
                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<half>(const size_t size, const half *input, half *output,
                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<float>(const size_t size, const float *input, float *output,
                                    const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<double>(const size_t size, const double *input, double *output,
                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<Complex<float>>(const size_t size, const Complex<float> *input,
                                             Complex<float> *output, const uint32_t &device_id,
                                             cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSinc<Complex<double>>(const size_t size, const Complex<double> *input,
                                              Complex<double> *output, const uint32_t &device_id,
                                              cudaStream_t cuda_stream);

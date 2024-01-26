/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/correlate_impl.cuh"

template <typename T>

__global__ void Conv1D(const T *input_addr, const T *kernel_addr, T *output_addr, const size_t out_size,
                       const size_t kernel_size, const size_t legal_st, const size_t legal_end) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < out_size; pos += blockDim.x * gridDim.x) {
    T temp = static_cast<T>(0);
    for (size_t i = 0; i < kernel_size; i++) {
      if (pos + i >= legal_st && pos + i < legal_end) {
        temp += input_addr[pos + i] * kernel_addr[i];
      }
    }
    output_addr[pos] = temp;
  }
}

template <typename T>
__global__ void Conj(const T *input_addr, T *output_addr, const size_t input_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_size; pos += blockDim.x * gridDim.x) {
    T temp_complex = T(input_addr[pos].real(), -input_addr[pos].imag());
    output_addr[pos] = temp_complex;
  }
}

template <typename T>
__global__ void Reverse(const T *input_addr, T *output_addr, const size_t *input_size_d, size_t input_size_h) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_size_h; pos += blockDim.x * gridDim.x) {
    size_t out_idx = input_size_d[0] - pos - 1;
    output_addr[out_idx] = input_addr[pos];
  }
}

template <typename T>
cudaError_t CorrelateCalc(const T *input_addr, const T *kernel_addr, T *output_addr, const size_t input_size,
                          const size_t kernel_size, const int64_t mode, const uint32_t &device_id,
                          cudaStream_t stream) {
  size_t out_size = input_size - kernel_size + 1;
  size_t legal_st = 0;
  size_t legal_end = 0;
  constexpr int64_t samemode = 1;
  constexpr int64_t validmode = 2;
  constexpr int64_t fullmode = 3;

  if (mode == samemode) {
    legal_st = kernel_size / 2;
    legal_end = input_size - kernel_size + kernel_size / 2 + 1;
  } else if (mode == validmode) {
    legal_st = 0;
    legal_end = input_size;
  } else if (mode == fullmode) {
    legal_st = kernel_size - 1;
    legal_end = out_size;
  }
  Conv1D<<<CUDA_BLOCKS(device_id, out_size), CUDA_THREADS(device_id), 0, stream>>>(
    input_addr, kernel_addr, output_addr, out_size, kernel_size, legal_st, legal_end);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalConj(const T *input_addr, T *output_addr, const size_t input_size, const uint32_t &device_id,
                    cudaStream_t stream) {
  Conj<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, stream>>>(input_addr, output_addr, input_size);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalReverse1D(const T *input_addr, T *output_addr, const size_t *input_size_d, size_t input_size_h,
                         const uint32_t &device_id, cudaStream_t stream) {
  Reverse<<<CUDA_BLOCKS(device_id, input_size_h), CUDA_THREADS(device_id), 0, stream>>>(input_addr, output_addr,
                                                                                        input_size_d, input_size_h);
  return GetCudaStatus();
}
template CUDA_LIB_EXPORT cudaError_t CorrelateCalc(const float *input_addr, const float *kernel_addr,
                                                   float *output_addr, const size_t input_size,
                                                   const size_t kernel_size, const int64_t mode,
                                                   const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CorrelateCalc(const double *input_addr, const double *kernel_addr,
                                                   double *output_addr, const size_t input_size,
                                                   const size_t kernel_size, const int64_t mode,
                                                   const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CorrelateCalc(const half *input_addr, const half *kernel_addr, half *output_addr,
                                                   const size_t input_size, const size_t kernel_size,
                                                   const int64_t mode, const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CorrelateCalc(const Complex<float> *input_addr, const Complex<float> *kernel_addr,
                                                   Complex<float> *output_addr, const size_t input_size,
                                                   const size_t kernel_size, const int64_t mode,
                                                   const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CorrelateCalc(const Complex<double> *input_addr,
                                                   const Complex<double> *kernel_addr, Complex<double> *output_addr,
                                                   const size_t input_size, const size_t kernel_size,
                                                   const int64_t mode, const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CalConj(const Complex<float> *input_addr, Complex<float> *output_addr,
                                             const size_t input_size, const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CalConj(const Complex<double> *input_addr, Complex<double> *output_addr,
                                             const size_t input_size, const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CalReverse1D(const Complex<float> *input_addr, Complex<float> *output_addr,
                                                  const size_t *input_size_d, size_t input_size_h,
                                                  const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CalReverse1D(const Complex<double> *input_addr, Complex<double> *output_addr,
                                                  const size_t *input_size_d, size_t input_size_h,
                                                  const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CalReverse1D(const float *input_addr, float *output_addr,
                                                  const size_t *input_size_d, size_t input_size_h,
                                                  const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CalReverse1D(const double *input_addr, double *output_addr,
                                                  const size_t *input_size_d, size_t input_size_h,
                                                  const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CalReverse1D(const half *input_addr, half *output_addr, const size_t *input_size_d,
                                                  size_t input_size_h, const uint32_t &device_id, cudaStream_t stream);

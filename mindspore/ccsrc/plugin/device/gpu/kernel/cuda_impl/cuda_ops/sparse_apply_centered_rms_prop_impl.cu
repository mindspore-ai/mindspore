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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_apply_centered_rms_prop_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T RsqrtFunc(T x) {
  return __frsqrt_rn(x);
}

template <>
__device__ __forceinline__ half RsqrtFunc(half x) {
  return hrsqrt(x);
}

template <>
__device__ __forceinline__ double RsqrtFunc(double x) {
  return rsqrt(x);
}

template <typename T, typename S>
__global__ void SparseApplyCenteredRMSPropUpdate(const size_t size, const size_t indices_size, const bool use_locking,
                                                 T *learning_rate, T *decay_rate, T *epsilon, T *momentum,
                                                 const T *gradient, const S *indices, T *variable, T *mean_grad,
                                                 T *mean_square, T *mom, T *variable_out) {
  const int64_t inner_size = static_cast<int64_t>(size * sizeof(int64_t) / sizeof(S));
  const T con1 = static_cast<T>(1);
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < static_cast<int64_t>(size);
       pos += gridDim.x * blockDim.x) {
    const int64_t index = pos / inner_size;
    const int64_t inner_pos = pos % inner_size;
    const int64_t grad_pos = pos;
    const int64_t cur_pos = indices[index] * inner_size + inner_pos;

    mean_square[cur_pos] =
      (*decay_rate) * mean_square[cur_pos] + (con1 - (*decay_rate)) * gradient[grad_pos] * gradient[grad_pos];
    mean_grad[cur_pos] = mean_grad[cur_pos] * (*decay_rate) + gradient[grad_pos] * (con1 - (*decay_rate));
    const T denom = mean_square[cur_pos] + (*epsilon) - mean_grad[cur_pos] * mean_grad[cur_pos];
    mom[cur_pos] = (*learning_rate) * gradient[grad_pos] * RsqrtFunc(denom) + mom[cur_pos] * (*momentum);
    variable_out[cur_pos] = variable[cur_pos] - mom[cur_pos];
  }
}
template <typename S>
__global__ void SparseApplyCenteredRMSPropUpdate(const size_t size, const size_t indices_size, const bool use_locking,
                                                 double *learning_rate, double *decay_rate, double *epsilon,
                                                 double *momentum, const double *gradient, const S *indices,
                                                 double *variable, double *mean_grad, double *mean_square, double *mom,
                                                 double *variable_out) {
  const int64_t inner_size = static_cast<int64_t>(size * sizeof(int64_t) / sizeof(S));
  const double con1 = static_cast<double>(1);
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < static_cast<int64_t>(size);
       pos += gridDim.x * blockDim.x) {
    const int64_t index = pos / inner_size;
    const int64_t inner_pos = pos % inner_size;
    const int64_t grad_pos = pos;
    const int64_t cur_pos = indices[index] * inner_size + inner_pos;

    mean_square[cur_pos] =
      (*decay_rate) * mean_square[cur_pos] + (con1 - (*decay_rate)) * gradient[grad_pos] * gradient[grad_pos];
    mean_grad[cur_pos] = mean_grad[cur_pos] * (*decay_rate) + gradient[grad_pos] * (con1 - (*decay_rate));
    const double denom = mean_square[cur_pos] + (*epsilon) - mean_grad[cur_pos] * mean_grad[cur_pos];
    mom[cur_pos] = (*learning_rate) * gradient[grad_pos] * RsqrtFunc(denom) + mom[cur_pos] * (*momentum);
    variable_out[cur_pos] = variable[cur_pos] - mom[cur_pos];
  }
}
template <typename S>
__global__ void SparseApplyCenteredRMSPropUpdate(const size_t size, const size_t indices_size, const bool use_locking,
                                                 half *learning_rate, half *decay_rate, half *epsilon, half *momentum,
                                                 const half *gradient, const S *indices, half *variable,
                                                 half *mean_grad, half *mean_square, half *mom, half *variable_out) {
  // const int64_t inner_size = static_cast<int64_t>(size / indices_size);
  const int64_t inner_size = static_cast<int64_t>(size * sizeof(int64_t) / sizeof(S));
  const float con1 = static_cast<float>(1);
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < static_cast<int64_t>(size);
       pos += gridDim.x * blockDim.x) {
    const int64_t index = pos / inner_size;
    const int64_t inner_pos = pos % inner_size;
    const int64_t grad_pos = pos;
    const int64_t cur_pos = indices[index] * inner_size + inner_pos;

    mean_square[cur_pos] = static_cast<float>(*decay_rate) * static_cast<float>(mean_square[cur_pos]) +
                           static_cast<float>(con1 - static_cast<float>(*decay_rate)) *
                             static_cast<float>(gradient[grad_pos]) * static_cast<float>(gradient[grad_pos]);
    mean_grad[cur_pos] = static_cast<float>(mean_grad[cur_pos]) * static_cast<float>(*decay_rate) +
                         static_cast<float>(gradient[grad_pos]) * (con1 - static_cast<float>(*decay_rate));
    const float denom = static_cast<float>(mean_square[cur_pos]) + static_cast<float>(*epsilon) -
                        static_cast<float>(mean_grad[cur_pos]) * static_cast<float>(mean_grad[cur_pos]);
    mom[cur_pos] = static_cast<float>(*learning_rate) * static_cast<float>(gradient[grad_pos]) *
                     static_cast<float>(RsqrtFunc(denom)) +
                   static_cast<float>(mom[cur_pos]) * static_cast<float>(*momentum);
    variable_out[cur_pos] =
      static_cast<float>(static_cast<float>(variable[cur_pos]) - static_cast<float>(mom[cur_pos]));
  }
}

template <typename T, typename S>
cudaError_t CalSparseApplyCenteredRMSProp(const size_t size, const size_t indices_size, const bool use_locking,
                                          T *learning_rate, T *decay_rate, T *epsilon, T *momentum, const T *gradient,
                                          const S *indices, T *variable, T *mean_grad, T *mean_square, T *mom,
                                          T *variable_out, cudaStream_t cuda_stream) {
  SparseApplyCenteredRMSPropUpdate<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    size, indices_size, use_locking, learning_rate, decay_rate, epsilon, momentum, gradient, indices, variable,
    mean_grad, mean_square, mom, variable_out);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<half, int32_t>(
  const size_t size, const size_t indices_size, const bool use_locking, half *learning_rate, half *decay_rate,
  half *epsilon, half *momentum, const half *gradient, const int32_t *indices, half *variable, half *mean_grad,
  half *mean_square, half *mom, half *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<float, int32_t>(
  const size_t size, const size_t indices_size, const bool use_locking, float *learning_rate, float *decay_rate,
  float *epsilon, float *momentum, const float *gradient, const int32_t *indices, float *variable, float *mean_grad,
  float *mean_square, float *mom, float *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<double, int32_t>(
  const size_t size, const size_t indices_size, const bool use_locking, double *learning_rate, double *decay_rate,
  double *epsilon, double *momentum, const double *gradient, const int32_t *indices, double *variable,
  double *mean_grad, double *mean_square, double *mom, double *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<int8_t, int32_t>(
  const size_t size, const size_t indices_size, const bool use_locking, int8_t *learning_rate, int8_t *decay_rate,
  int8_t *epsilon, int8_t *momentum, const int8_t *gradient, const int32_t *indices, int8_t *variable,
  int8_t *mean_grad, int8_t *mean_square, int8_t *mom, int8_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<int16_t, int32_t>(
  const size_t size, const size_t indices_size, const bool use_locking, int16_t *learning_rate, int16_t *decay_rate,
  int16_t *epsilon, int16_t *momentum, const int16_t *gradient, const int32_t *indices, int16_t *variable,
  int16_t *mean_grad, int16_t *mean_square, int16_t *mom, int16_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<int32_t, int32_t>(
  const size_t size, const size_t indices_size, const bool use_locking, int32_t *learning_rate, int32_t *decay_rate,
  int32_t *epsilon, int32_t *momentum, const int32_t *gradient, const int32_t *indices, int32_t *variable,
  int32_t *mean_grad, int32_t *mean_square, int32_t *mom, int32_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<int64_t, int32_t>(
  const size_t size, const size_t indices_size, const bool use_locking, int64_t *learning_rate, int64_t *decay_rate,
  int64_t *epsilon, int64_t *momentum, const int64_t *gradient, const int32_t *indices, int64_t *variable,
  int64_t *mean_grad, int64_t *mean_square, int64_t *mom, int64_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<uint8_t, int32_t>(
  const size_t size, const size_t indices_size, const bool use_locking, uint8_t *learning_rate, uint8_t *decay_rate,
  uint8_t *epsilon, uint8_t *momentum, const uint8_t *gradient, const int32_t *indices, uint8_t *variable,
  uint8_t *mean_grad, uint8_t *mean_square, uint8_t *mom, uint8_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<uint16_t, int32_t>(
  const size_t size, const size_t indices_size, const bool use_locking, uint16_t *learning_rate, uint16_t *decay_rate,
  uint16_t *epsilon, uint16_t *momentum, const uint16_t *gradient, const int32_t *indices, uint16_t *variable,
  uint16_t *mean_grad, uint16_t *mean_square, uint16_t *mom, uint16_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<uint32_t, int32_t>(
  const size_t size, const size_t indices_size, const bool use_locking, uint32_t *learning_rate, uint32_t *decay_rate,
  uint32_t *epsilon, uint32_t *momentum, const uint32_t *gradient, const int32_t *indices, uint32_t *variable,
  uint32_t *mean_grad, uint32_t *mean_square, uint32_t *mom, uint32_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<uint64_t, int32_t>(
  const size_t size, const size_t indices_size, const bool use_locking, uint64_t *learning_rate, uint64_t *decay_rate,
  uint64_t *epsilon, uint64_t *momentum, const uint64_t *gradient, const int32_t *indices, uint64_t *variable,
  uint64_t *mean_grad, uint64_t *mean_square, uint64_t *mom, uint64_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<half, int64_t>(
  const size_t size, const size_t indices_size, const bool use_locking, half *learning_rate, half *decay_rate,
  half *epsilon, half *momentum, const half *gradient, const int64_t *indices, half *variable, half *mean_grad,
  half *mean_square, half *mom, half *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<float, int64_t>(
  const size_t size, const size_t indices_size, const bool use_locking, float *learning_rate, float *decay_rate,
  float *epsilon, float *momentum, const float *gradient, const int64_t *indices, float *variable, float *mean_grad,
  float *mean_square, float *mom, float *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<double, int64_t>(
  const size_t size, const size_t indices_size, const bool use_locking, double *learning_rate, double *decay_rate,
  double *epsilon, double *momentum, const double *gradient, const int64_t *indices, double *variable,
  double *mean_grad, double *mean_square, double *mom, double *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<int8_t, int64_t>(
  const size_t size, const size_t indices_size, const bool use_locking, int8_t *learning_rate, int8_t *decay_rate,
  int8_t *epsilon, int8_t *momentum, const int8_t *gradient, const int64_t *indices, int8_t *variable,
  int8_t *mean_grad, int8_t *mean_square, int8_t *mom, int8_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<int16_t, int64_t>(
  const size_t size, const size_t indices_size, const bool use_locking, int16_t *learning_rate, int16_t *decay_rate,
  int16_t *epsilon, int16_t *momentum, const int16_t *gradient, const int64_t *indices, int16_t *variable,
  int16_t *mean_grad, int16_t *mean_square, int16_t *mom, int16_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<int32_t, int64_t>(
  const size_t size, const size_t indices_size, const bool use_locking, int32_t *learning_rate, int32_t *decay_rate,
  int32_t *epsilon, int32_t *momentum, const int32_t *gradient, const int64_t *indices, int32_t *variable,
  int32_t *mean_grad, int32_t *mean_square, int32_t *mom, int32_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<int64_t, int64_t>(
  const size_t size, const size_t indices_size, const bool use_locking, int64_t *learning_rate, int64_t *decay_rate,
  int64_t *epsilon, int64_t *momentum, const int64_t *gradient, const int64_t *indices, int64_t *variable,
  int64_t *mean_grad, int64_t *mean_square, int64_t *mom, int64_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<uint8_t, int64_t>(
  const size_t size, const size_t indices_size, const bool use_locking, uint8_t *learning_rate, uint8_t *decay_rate,
  uint8_t *epsilon, uint8_t *momentum, const uint8_t *gradient, const int64_t *indices, uint8_t *variable,
  uint8_t *mean_grad, uint8_t *mean_square, uint8_t *mom, uint8_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<uint16_t, int64_t>(
  const size_t size, const size_t indices_size, const bool use_locking, uint16_t *learning_rate, uint16_t *decay_rate,
  uint16_t *epsilon, uint16_t *momentum, const uint16_t *gradient, const int64_t *indices, uint16_t *variable,
  uint16_t *mean_grad, uint16_t *mean_square, uint16_t *mom, uint16_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<uint32_t, int64_t>(
  const size_t size, const size_t indices_size, const bool use_locking, uint32_t *learning_rate, uint32_t *decay_rate,
  uint32_t *epsilon, uint32_t *momentum, const uint32_t *gradient, const int64_t *indices, uint32_t *variable,
  uint32_t *mean_grad, uint32_t *mean_square, uint32_t *mom, uint32_t *variable_out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyCenteredRMSProp<uint64_t, int64_t>(
  const size_t size, const size_t indices_size, const bool use_locking, uint64_t *learning_rate, uint64_t *decay_rate,
  uint64_t *epsilon, uint64_t *momentum, const uint64_t *gradient, const int64_t *indices, uint64_t *variable,
  uint64_t *mean_grad, uint64_t *mean_square, uint64_t *mom, uint64_t *variable_out, cudaStream_t cuda_stream);

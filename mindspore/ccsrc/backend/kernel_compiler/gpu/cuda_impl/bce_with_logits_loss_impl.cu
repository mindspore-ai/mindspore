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

#include "backend/kernel_compiler/gpu/cuda_impl/bce_with_logits_loss_impl.cuh"

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T>
__global__ void FillWithoutBroadcast(const size_t size, const T *src, T *dst) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    dst[pos] = src[pos];
  }
  return;
}

template <typename T>
__global__ void FillAndBroadcast(const size_t size, const size_t shape_size, const size_t *src_shape,
                                 const size_t *dst_shape, const T *src, T *dst) {
  size_t dst_index_array[MAX_LOGITS_DIMENSION];
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    size_t tmp_pos = pos;
    size_t pos_size = size / dst_shape[0];
    dst_index_array[0] = tmp_pos / pos_size;
    for (size_t i = 1; i < shape_size; i++) {
      tmp_pos -= dst_index_array[i - 1] * pos_size;
      pos_size = pos_size / dst_shape[i];
      dst_index_array[i] = tmp_pos / pos_size;
    }
    size_t src_pos = 0;
    size_t src_size = 1;
    for (size_t i = 0; i < shape_size; i++) {
      src_size *= src_shape[i];
    }
    for (size_t i = 0; i < shape_size; i++) {
      src_size /= src_shape[i];
      size_t length_by_index = Index(dst_index_array[i], src_shape[i]) * src_size;
      src_pos += length_by_index;
    }
    dst[pos] = src[src_pos];
  }
  return;
}

template <typename T>
__global__ void BCEWithLogitsLossMain(size_t size, const T *predict, const T *target, const T *shape_broadcasted,
                                      T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T max_value = -predict[pos];
    max_value = max_value > static_cast<T>(0) ? max_value : static_cast<T>(0);
    const T log_weight = (shape_broadcasted[pos] - static_cast<T>(1)) * target[pos] + static_cast<T>(1);
    output[pos] = (static_cast<T>(1) - target[pos]) * predict[pos] +
                  log_weight * (log(exp(-max_value) + exp(-predict[pos] - max_value)) + max_value);
  }
  return;
}

template <>
__global__ void BCEWithLogitsLossMain(size_t size, const half *predict, const half *target,
                                      const half *shape_broadcasted, half *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    half max_value = -predict[pos];
    max_value = max_value > static_cast<half>(0) ? max_value : static_cast<half>(0);
    const half log_weight = (shape_broadcasted[pos] - static_cast<half>(1)) * target[pos] + static_cast<half>(1);
    output[pos] = (static_cast<half>(1) - target[pos]) * predict[pos] +
                  log_weight * (hlog(hexp(-max_value) + hexp(-predict[pos] - max_value)) + max_value);
  }
  return;
}

template <typename T>
__global__ void Mul(size_t size, const T *lhs, T *rhs) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    rhs[pos] *= lhs[pos];
  }
  return;
}

template <typename T>
void CalBCEWithLogitsLoss(const size_t input_size, const T *predict, const T *target, const size_t *input_shape,
                          const size_t shape_size, const T *weight, const size_t *weight_shape,
                          const bool weight_need_broadcast, const T *pos_weight, const size_t *pos_weight_shape,
                          const bool pos_weight_need_broadcast, T *shape_broadcasted, T *output,
                          cudaStream_t cuda_stream) {
  if (pos_weight_need_broadcast) {
    FillAndBroadcast<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(
      input_size, shape_size, pos_weight_shape, input_shape, pos_weight, shape_broadcasted);
  } else {
    FillWithoutBroadcast<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input_size, pos_weight,
                                                                                  shape_broadcasted);
  }
  BCEWithLogitsLossMain<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input_size, predict, target,
                                                                                 shape_broadcasted, output);
  if (weight_need_broadcast) {
    FillAndBroadcast<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input_size, shape_size, weight_shape,
                                                                              input_shape, weight, shape_broadcasted);
  } else {
    FillWithoutBroadcast<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input_size, weight,
                                                                                  shape_broadcasted);
  }
  Mul<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input_size, shape_broadcasted, output);
  return;
}

template void CalBCEWithLogitsLoss<half>(const size_t input_size, const half *predict, const half *target,
                                         const size_t *input_shape, const size_t shape_size, const half *weight,
                                         const size_t *weight_shape, const bool weight_need_broadcast,
                                         const half *pos_weight, const size_t *pos_weight_shape,
                                         const bool pos_weight_need_broadcast, half *shape_broadcasted, half *output,
                                         cudaStream_t cuda_stream);
template void CalBCEWithLogitsLoss<float>(const size_t input_size, const float *predict, const float *target,
                                          const size_t *input_shape, const size_t shape_size, const float *weight,
                                          const size_t *weight_shape, const bool weight_need_broadcast,
                                          const float *pos_weight, const size_t *pos_weight_shape,
                                          const bool pos_weight_need_broadcast, float *shape_broadcasted, float *output,
                                          cudaStream_t cuda_stream);

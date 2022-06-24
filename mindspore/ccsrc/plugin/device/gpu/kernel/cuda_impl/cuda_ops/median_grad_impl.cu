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

#include "median_grad_impl.cuh"
#include <iostream>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void Count_Repeat(const T *x, const T *y, int64_t size, int *repeat_val) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    if (x[pos] == *y) {
      MsAtomicAdd(repeat_val, 1);
    }
  }
}

template <typename T, typename V>
__global__ void GlobalMedianGradComputer(const T *y_grad, const T *x, const T *y, V *output, int *repeat_val,
                                         const int64_t size) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    if (x[pos] == *y) {
      output[pos] = *y_grad / *repeat_val;
    } else {
      output[pos] = 0;
    }
  }
}

template <typename T, typename S, typename V>
__global__ void MedianGradComputer(const T *y_grad, const S *indices, const T *y, V *output, int *elem_num_each_dim_x,
                                   int *elem_num_each_dim_y, int64_t axis, int64_t input_dim, int64_t size) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int elements_remain = pos;
    int temp = 0;
    int update_pos = 0;
    for (int i = 0; i < input_dim; i++) {
      temp = elements_remain / elem_num_each_dim_y[i];
      elements_remain %= elem_num_each_dim_y[i];
      if (i == axis) {
        update_pos += *(indices + pos) * elem_num_each_dim_x[i];
      } else {
        update_pos += temp * elem_num_each_dim_x[i];
      }
    }
    *(output + update_pos) = *(y_grad + pos);
  }
}

template <typename T, typename S, typename V>
void MedianGrad(const T *y_grad, const T *x, const T *y, const S *indices, V *output, const int64_t axis,
                bool global_median, const int64_t input0_size, const int64_t input1_size, int64_t input_dim,
                int *elem_num_each_dim_x, int *elem_num_each_dim_y, int *repeat_val, cudaStream_t cuda_stream) {
  if (global_median) {
    Count_Repeat<T><<<GET_BLOCKS(input1_size), GET_THREADS, 0, cuda_stream>>>(x, y, input1_size, repeat_val);
    GlobalMedianGradComputer<T, V>
      <<<GET_BLOCKS(input1_size), GET_THREADS, 0, cuda_stream>>>(y_grad, x, y, output, repeat_val, input1_size);
  } else {
    MedianGradComputer<T, S, V><<<GET_BLOCKS(input0_size), GET_THREADS, 0, cuda_stream>>>(
      y_grad, indices, y, output, elem_num_each_dim_x, elem_num_each_dim_y, axis, input_dim, input0_size);
  }
}

template CUDA_LIB_EXPORT void MedianGrad<int16_t, int64_t, float>(
  const int16_t *input0_value, const int16_t *input1_value, const int16_t *input2_value, const int64_t *input3_value,
  float *output, const int64_t axis, bool global_median, const int64_t input0_size, const int64_t input1_size,
  int64_t input_dim, int *elem_num_each_dim_x, int *elem_num_each_dim_y, int *repeat_val, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MedianGrad<int32_t, int64_t, float>(
  const int32_t *input0_value, const int32_t *input1_value, const int32_t *input2_value, const int64_t *input3_value,
  float *output, const int64_t axis, bool global_median, const int64_t input0_size, const int64_t input1_size,
  int64_t input_dim, int *elem_num_each_dim_x, int *elem_num_each_dim_y, int *repeat_val, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MedianGrad<int64_t, int64_t, float>(
  const int64_t *input0_value, const int64_t *input1_value, const int64_t *input2_value, const int64_t *input3_value,
  float *output, const int64_t axis, bool global_median, const int64_t input0_size, const int64_t input1_size,
  int64_t input_dim, int *elem_num_each_dim_x, int *elem_num_each_dim_y, int *repeat_val, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MedianGrad<float, int64_t, float>(
  const float *input0_value, const float *input1_value, const float *input2_value, const int64_t *input3_value,
  float *output, const int64_t axis, bool global_median, const int64_t input0_size, const int64_t input1_size,
  int64_t input_dim, int *elem_num_each_dim_x, int *elem_num_each_dim_y, int *repeat_val, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MedianGrad<double, int64_t, double>(
  const double *input0_value, const double *input1_value, const double *input2_value, const int64_t *input3_value,
  double *output, const int64_t axis, bool global_median, const int64_t input0_size, const int64_t input1_size,
  int64_t input_dim, int *elem_num_each_dim_x, int *elem_num_each_dim_y, int *repeat_val, cudaStream_t cuda_stream);

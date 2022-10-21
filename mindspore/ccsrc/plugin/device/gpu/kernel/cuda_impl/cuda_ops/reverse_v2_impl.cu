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
#include <cuda_runtime.h>
#include "reverse_v2_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void ReverseV2(const T* input, T* output, const size_t* input_shape, const int64_t* strides,
                          const int64_t* axis, size_t input_size, size_t axis_size) {
  for (int64_t gt_id = blockIdx.x * blockDim.x + threadIdx.x; gt_id < input_size; gt_id += blockDim.x * gridDim.x) {
    int64_t intermediate_index = gt_id;
    for (size_t i = 0; i < axis_size; i++) {
      int64_t d = axis[i];
      int64_t pre_reverse_position = (gt_id / strides[d]) % input_shape[d];
      int64_t reversed_position = input_shape[d] - pre_reverse_position - 1;
      intermediate_index += ((reversed_position - pre_reverse_position) * strides[d]);
    }

    output[intermediate_index] = input[gt_id];
  }
  return;
}
template <typename T>
void CalReverseV2(const T* input, T* output, const size_t* input_shape, const int64_t* strides, const int64_t* axis,
                  size_t input_size, size_t axis_size, cudaStream_t cuda_stream) {
  ReverseV2<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input, output, input_shape, strides, axis,
                                                                     input_size, axis_size);
  return;
}

template CUDA_LIB_EXPORT void CalReverseV2<Complex<float>>(const Complex<float>* input, Complex<float>* output,
                                                 const size_t* input_shape, const int64_t* strides,
                                                 const int64_t* axis, size_t input_size,
                                                 size_t axis_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReverseV2<Complex<double>>(const Complex<double>* input, Complex<double>* output,
                                                 const size_t* input_shape, const int64_t* strides,
                                                 const int64_t* axis, size_t input_size,
                                                 size_t axis_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReverseV2<half>(const half* input, half* output, const size_t* input_shape,
                                                 const int64_t* strides, const int64_t* axis, size_t input_size,
                                                 size_t axis_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReverseV2<float>(const float* input, float* output, const size_t* input_shape,
                                                  const int64_t* strides, const int64_t* axis, size_t input_size,
                                                  size_t axis_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReverseV2<double>(const double* input, double* output, const size_t* input_shape,
                                                  const int64_t* strides, const int64_t* axis, size_t input_size,
                                                  size_t axis_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReverseV2<uint8_t>(const uint8_t* input, uint8_t* output, const size_t* input_shape,
                                                    const int64_t* strides, const int64_t* axis, size_t input_size,
                                                    size_t axis_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReverseV2<uint16_t>(const uint16_t* input, uint16_t* output, const size_t* input_shape,
                                                    const int64_t* strides, const int64_t* axis, size_t input_size,
                                                    size_t axis_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReverseV2<int8_t>(const int8_t* input, int8_t* output, const size_t* input_shape,
                                                    const int64_t* strides, const int64_t* axis, size_t input_size,
                                                    size_t axis_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReverseV2<int16_t>(const int16_t* input, int16_t* output, const size_t* input_shape,
                                                    const int64_t* strides, const int64_t* axis, size_t input_size,
                                                    size_t axis_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReverseV2<int32_t>(const int32_t* input, int32_t* output, const size_t* input_shape,
                                                    const int64_t* strides, const int64_t* axis, size_t input_size,
                                                    size_t axis_size, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReverseV2<int64_t>(const int64_t* input, int64_t* output, const size_t* input_shape,
                                                    const int64_t* strides, const int64_t* axis, size_t input_size,
                                                    size_t axis_size, cudaStream_t cuda_stream);

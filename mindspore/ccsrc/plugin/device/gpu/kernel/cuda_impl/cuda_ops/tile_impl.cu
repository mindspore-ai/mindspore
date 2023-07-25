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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tile_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

struct StrideInfo {
  size_t inp_stride[TILE_MAX_DIMENSION + 1];
  size_t out_stride[TILE_MAX_DIMENSION + 1];
};

template <typename T>
using Complex = mindspore::utils::Complex<T>;
template <typename T>
__global__ void Tile(const size_t output_size, const size_t shape_size, const StrideInfo strides, const T *input,
                     T *output) {
  // for example 4-D: pos = pos_array[0] * output_shape[1] * output_shape[2] * output_shape[3] +
  //                        pos_array[1] * output_shape[2] * output_shape[3] +
  //                        pos_array[2] * output_shape[3] +
  //                        pos_array[3]
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_size; pos += blockDim.x * gridDim.x) {
    size_t tmp_pos = pos;
    size_t cur_idx = 0;
    size_t input_pos = 0;
    size_t cur_inp_shape = 0;
    for (size_t idx = 0; idx < shape_size; ++idx) {
      cur_idx = tmp_pos / strides.out_stride[idx + 1];
      cur_inp_shape = strides.inp_stride[idx] / strides.inp_stride[idx + 1];
      input_pos += (cur_idx % cur_inp_shape) * strides.inp_stride[idx + 1];
      tmp_pos -= cur_idx * strides.out_stride[idx + 1];
    }
    output[pos] = input[input_pos];
  }
}

StrideInfo CalStride(const std::vector<size_t> inp_shape, const std::vector<size_t> out_shape) {
  StrideInfo strides;
  if (out_shape.size() == 0) {
    strides.inp_stride[0] = 1;
    strides.inp_stride[1] = 1;
    strides.out_stride[0] = 1;
    strides.out_stride[1] = 1;
    return strides;
  }
  strides.inp_stride[out_shape.size()] = 1;
  strides.out_stride[out_shape.size()] = 1;
  for (int idx = out_shape.size() - 1; idx >= 0; --idx) {
    strides.inp_stride[idx] = strides.inp_stride[idx + 1] * inp_shape[idx];
    strides.out_stride[idx] = strides.out_stride[idx + 1] * out_shape[idx];
  }
  return strides;
}
template <typename T>
cudaError_t CalTile(const std::vector<size_t> inp_shape, const std::vector<size_t> out_shape, const T *input, T *output,
                    cudaStream_t cuda_stream) {
  StrideInfo strides = CalStride(inp_shape, out_shape);
  size_t output_size = 1;
  for (auto val : out_shape) {
    output_size *= val;
  }
  size_t thread_num = output_size > 512 ? 512 : output_size;
  Tile<<<CUDA_BLOCKS_CAL(GET_CTX_DEVICE_ID, output_size, thread_num), thread_num, 0, cuda_stream>>>(
    output_size, out_shape.size(), strides, input, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalTile<Complex<float>>(const std::vector<size_t> inp_shape,
                                                             const std::vector<size_t> out_shape,
                                                             const Complex<float> *input, Complex<float> *output,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTile<Complex<double>>(const std::vector<size_t> inp_shape,
                                                              const std::vector<size_t> out_shape,
                                                              const Complex<double> *input, Complex<double> *output,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTile<double>(const std::vector<size_t> inp_shape,
                                                     const std::vector<size_t> out_shape, const double *input,
                                                     double *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTile<float>(const std::vector<size_t> inp_shape,
                                                    const std::vector<size_t> out_shape, const float *input,
                                                    float *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTile<half>(const std::vector<size_t> inp_shape,
                                                   const std::vector<size_t> out_shape, const half *input, half *output,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTile<int16_t>(const std::vector<size_t> inp_shape,
                                                      const std::vector<size_t> out_shape, const int16_t *input,
                                                      int16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTile<int>(const std::vector<size_t> inp_shape,
                                                  const std::vector<size_t> out_shape, const int *input, int *output,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTile<int64_t>(const std::vector<size_t> inp_shape,
                                                      const std::vector<size_t> out_shape, const int64_t *input,
                                                      int64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTile<bool>(const std::vector<size_t> inp_shape,
                                                   const std::vector<size_t> out_shape, const bool *input, bool *output,
                                                   cudaStream_t cuda_stream);

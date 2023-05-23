/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITH WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_to_impl.cuh"
#include <math.h>
#include <vector>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
struct UnaryBroadcastStrideInfo {
  size_t input_stride[8];
  size_t output_stride[8];
};
// copy
template <typename T>
__global__ void BroadcastToCpyCuda(size_t dim_size, size_t output_num, UnaryBroadcastStrideInfo strides, T *input,
                                   T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_num; pos += blockDim.x * gridDim.x) {
    int64_t cur_out_idx = 0;
    size_t cur_pos = pos;
    size_t inp_pos = 0;
    for (int idx = 0; idx < dim_size; ++idx) {
      cur_out_idx = cur_pos / strides.output_stride[idx];
      inp_pos += cur_out_idx * strides.input_stride[idx];
      cur_pos -= cur_out_idx * strides.output_stride[idx];
    }
    output[pos] = input[inp_pos];
  }
}

template <typename T>
cudaError_t BroadcastTo(const std::vector<int64_t> &inp_shape, const std::vector<int64_t> &out_shape, T *input,
                        T *output, size_t device_id, cudaStream_t cuda_stream) {
  const size_t dim_size = out_shape.size();
  UnaryBroadcastStrideInfo strides;
  size_t output_num = out_shape.back();
  strides.input_stride[dim_size - 1] = 1;
  strides.output_stride[dim_size - 1] = 1;
  for (int64_t idx = dim_size - 2; idx >= 0; --idx) {
    strides.output_stride[idx] = out_shape[idx + 1] * strides.output_stride[idx + 1];
    strides.input_stride[idx] = inp_shape[idx + 1] * strides.input_stride[idx + 1];
    output_num *= out_shape[idx];
  }
  for (size_t idx = 0; idx < dim_size; ++idx) {
    strides.input_stride[idx] = (inp_shape[idx] == 1) ? 0 : strides.input_stride[idx];
  }
  size_t thread_num = output_num > 1024 ? 1024 : output_num;
  BroadcastToCpyCuda<T><<<CUDA_BLOCKS_CAL(device_id, output_num, thread_num), thread_num, 0, cuda_stream>>>(
    dim_size, output_num, strides, input, output);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t BroadcastTo<bool>(const std::vector<int64_t> &inp_shape,
                                                       const std::vector<int64_t> &out_shape, bool *input, bool *output,
                                                       size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<int8_t>(const std::vector<int64_t> &inp_shape,
                                                         const std::vector<int64_t> &out_shape, int8_t *input,
                                                         int8_t *output, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<int16_t>(const std::vector<int64_t> &inp_shape,
                                                          const std::vector<int64_t> &out_shape, int16_t *input,
                                                          int16_t *output, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<int32_t>(const std::vector<int64_t> &inp_shape,
                                                          const std::vector<int64_t> &out_shape, int32_t *input,
                                                          int32_t *output, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<int64_t>(const std::vector<int64_t> &inp_shape,
                                                          const std::vector<int64_t> &out_shape, int64_t *input,
                                                          int64_t *output, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<uint8_t>(const std::vector<int64_t> &inp_shape,
                                                          const std::vector<int64_t> &out_shape, uint8_t *input,
                                                          uint8_t *output, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<uint16_t>(const std::vector<int64_t> &inp_shape,
                                                           const std::vector<int64_t> &out_shape, uint16_t *input,
                                                           uint16_t *output, size_t device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<uint32_t>(const std::vector<int64_t> &inp_shape,
                                                           const std::vector<int64_t> &out_shape, uint32_t *input,
                                                           uint32_t *output, size_t device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<uint64_t>(const std::vector<int64_t> &inp_shape,
                                                           const std::vector<int64_t> &out_shape, uint64_t *input,
                                                           uint64_t *output, size_t device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<half>(const std::vector<int64_t> &inp_shape,
                                                       const std::vector<int64_t> &out_shape, half *input, half *output,
                                                       size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<float>(const std::vector<int64_t> &inp_shape,
                                                        const std::vector<int64_t> &out_shape, float *input,
                                                        float *output, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<double>(const std::vector<int64_t> &inp_shape,
                                                         const std::vector<int64_t> &out_shape, double *input,
                                                         double *output, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<Complex<float>>(const std::vector<int64_t> &inp_shape,
                                                                 const std::vector<int64_t> &out_shape,
                                                                 Complex<float> *input, Complex<float> *output,
                                                                 size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastTo<Complex<double>>(const std::vector<int64_t> &inp_shape,
                                                                  const std::vector<int64_t> &out_shape,
                                                                  Complex<double> *input, Complex<double> *output,
                                                                  size_t device_id, cudaStream_t cuda_stream);

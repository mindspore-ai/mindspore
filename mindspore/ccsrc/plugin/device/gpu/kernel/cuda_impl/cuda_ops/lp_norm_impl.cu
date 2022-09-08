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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/lp_norm_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__inline__ __device__ void LpNormCoreOp(const T *input, size_t input_index, float *output, size_t output_index,
                                        float p) {
  T abs_value = input[input_index] >= static_cast<T>(0) ? input[input_index] : -input[input_index];
  // We do parallel LpNorm by input elements. So multiple input data will be reduce sum to output, which causes data
  // competition.
  (void)MsAtomicAdd(output + output_index, pow(abs_value, p));
}

template <typename T>
__global__ void LpCalKernel(const T *input, const size_t *input_shape, size_t input_shape_length, size_t input_elements,
                            const size_t *output_axis, const size_t *output_stride, size_t output_shape_length, float p,
                            float eps, float *middle_output) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (input_elements);
       index += blockDim.x * gridDim.x) {
    size_t flatten_index = index;
    size_t output_index = 0;
    for (int i = static_cast<int>(input_shape_length - 1); i >= 0; --i) {
      for (int j = static_cast<int>(output_shape_length - 1); j >= 0; --j) {
        // 1. Calculate coordinate by input shape.
        size_t coordinate = flatten_index % input_shape[i];
        // 2. Map input coordinate to output coordinate by axis.
        if (i == output_axis[j]) {
          // 3. Calculate output index by output coordinate.
          output_index += coordinate * output_stride[j];
          break;
        }
      }
      flatten_index = flatten_index / input_shape[i];
    }
    LpNormCoreOp(input, index, middle_output, output_index, p);
  }
}

template <typename T>
__global__ void NormCalKernel(T *output, size_t output_elements, float p, float eps) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (output_elements);
       index += blockDim.x * gridDim.x) {
    output[index] = pow(output[index], 1 / p);
  }
}

template <typename T>
__global__ void NormCalHighPrecisionKernel(const float *middle_output, T *output, size_t output_elements, float p,
                                           float eps) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (output_elements);
       index += blockDim.x * gridDim.x) {
    output[index] = pow(middle_output[index], 1 / p);
  }
}

template <>
void CalLpNorm<float>(const float *input, const size_t *input_shape, size_t input_shape_length,
                      size_t input_elements, const size_t *output_axis, const size_t *output_stride,
                      size_t output_shape_length, size_t output_elements, float p, float eps,
                      float *middle_output, float *output, const uint32_t &device_id,
                      cudaStream_t cuda_stream) {
  LpCalKernel<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, input_shape, input_shape_length, input_elements, output_axis, output_stride, output_shape_length, p, eps,
    output);
  NormCalKernel<<<CUDA_BLOCKS(device_id, output_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    output, output_elements, p, eps);
}

template <>
void CalLpNorm<half>(const half *input, const size_t *input_shape, size_t input_shape_length,
                     size_t input_elements, const size_t *output_axis, const size_t *output_stride,
                     size_t output_shape_length, size_t output_elements, float p, float eps,
                     float *middle_output, half *output, const uint32_t &device_id,
                     cudaStream_t cuda_stream) {
  LpCalKernel<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, input_shape, input_shape_length, input_elements, output_axis, output_stride, output_shape_length, p, eps,
    middle_output);
  NormCalHighPrecisionKernel<<<CUDA_BLOCKS(device_id, output_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    middle_output, output, output_elements, p, eps);
}

template CUDA_LIB_EXPORT
void CalLpNorm<float>(const float *input, const size_t *input_shape, size_t input_shape_length,
                      size_t input_elements, const size_t *output_axis, const size_t *output_stride,
                      size_t output_shape_length, size_t output_elements, float p, float eps,
                      float *middle_output, float *output, const uint32_t &device_id,
                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT
void CalLpNorm<half>(const half *input, const size_t *input_shape, size_t input_shape_length,
                     size_t input_elements, const size_t *output_axis, const size_t *output_stride,
                     size_t output_shape_length, size_t output_elements, float p, float eps,
                     float *middle_output, half *output, const uint32_t &device_id,
                     cudaStream_t cuda_stream);

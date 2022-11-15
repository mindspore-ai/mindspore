/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, softwareg
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <iostream>
#include <limits>

#include "adjust_contrast_v2_impl.cuh"

template <typename T>
__global__ void AdjustContrastV2GpuKernel(const T* images, const float* contrast_factor, T* images_out, const int total,
                                          const int per_batch_elements) {
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < total; k += blockDim.x * gridDim.x) {
      float sum = 0;
      for (int i = 0; i < per_batch_elements; i += 3) {
        sum += static_cast<float>(images[k / 3 * per_batch_elements + i + k % 3]);
      }
      float mean = sum / (per_batch_elements / 3);
      for (int i = 0; i < per_batch_elements; i += 3) {
        images_out[k / 3 * per_batch_elements + i + k % 3] = static_cast<T>(
            (static_cast<float>(images[k / 3 * per_batch_elements + i + k % 3]) - mean) * contrast_factor[0] + mean);
      }
  }
}

template <typename T>
void CalAdjustContrastV2GpuKernel(const T* images, const float* contrast_factor, T* images_out, const int total,
                                  const int per_batch_elements, const uint32_t& device_id, cudaStream_t cuda_stream) {
  AdjustContrastV2GpuKernel<<<CUDA_BLOCKS(device_id, total), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      images, contrast_factor, images_out, total, per_batch_elements);
}

template CUDA_LIB_EXPORT void CalAdjustContrastV2GpuKernel<half>(const half* images, const float* contrast_factor,
                                                                 half* images_out, const int total,
                                                                 const int per_batch_elements,
                                                                 const uint32_t& device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalAdjustContrastV2GpuKernel<float>(const float* images, const float* contrast_factor,
                                                                  float* images_out, const int total,
                                                                  const int per_batch_elements,
                                                                  const uint32_t& device_id, cudaStream_t cuda_stream);

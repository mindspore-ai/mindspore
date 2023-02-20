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
  const int thread_num = 128;
  const int block_num = 128;
  for (int k = blockIdx.x; k < total; k += block_num) {
    __shared__ float ssum[thread_num];  // calculate the sum of elements dealt by the threads
    memset(ssum, 0, thread_num * sizeof(float));
    __syncthreads();
    float sum = 0;
    int base = k / 3 * per_batch_elements + k % 3;
    int group_num = per_batch_elements / thread_num;  // each thread deals with group_num elements
    while (group_num * thread_num < per_batch_elements || group_num % 3 != 0) {
      group_num++;  // make the number a multiple of 3
    }
    for (int stride = 0; stride < group_num; stride += 3) {
      if (stride + threadIdx.x * group_num < per_batch_elements) {
        atomicAdd(&ssum[threadIdx.x], images[base + stride + threadIdx.x * group_num]);
      }
    }
    __syncthreads();

    for (int i = 0; i < thread_num; i++) {
      sum += ssum[i];
    }

    float mean = sum / (per_batch_elements / 3);

    for (int i = threadIdx.x; i < per_batch_elements; i += thread_num) {
      if (i % 3 == 0) {
        images_out[base + i] = static_cast<T>((static_cast<float>(images[base + i]) - mean) * contrast_factor[0] +
                                               mean);
      }
    }
  }
}

template <typename T>
void CalAdjustContrastV2GpuKernel(const T* images, const float* contrast_factor, T* images_out, const int total,
                                  const int per_batch_elements, const uint32_t& device_id, cudaStream_t cuda_stream) {
  AdjustContrastV2GpuKernel<<<128, 128, 0, cuda_stream>>>(
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

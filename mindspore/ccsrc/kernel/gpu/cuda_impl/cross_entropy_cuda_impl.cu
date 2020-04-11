/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <stdint.h>
#include "cross_entropy_cuda_impl.cuh"
#include "include/cuda_runtime.h"

__global__ void CalCrossEntropyWithGradKernel(const float *softmax_logits, const float *log_softmax_logits,
                                              const float *labels, const int batch_size, const int num_classes,
                                              float *loss, float *dx) {
  extern __shared__ float loss_shared[];
  const float mean_scale = 1.0f / static_cast<float>(batch_size);

  loss_shared[threadIdx.x] = 0;
  for (int i = threadIdx.x * num_classes; i < (threadIdx.x + 1) * num_classes; ++i) {
    loss_shared[threadIdx.x] -= log_softmax_logits[i] * labels[i];
    dx[i] = (softmax_logits[i] - labels[i]) * mean_scale;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    *loss = 0;
    for (int i = 0; i < batch_size; i++) {
      *loss += loss_shared[i];
    }
    *loss *= mean_scale;
  }
}

void CalCrossEntropyWithGrad(const float *softmax_logits, const float *log_softmax_logits, const float *labels,
                             const int batch_size, const int num_classes, float *loss, float *dx,
                             cudaStream_t cuda_stream) {
  CalCrossEntropyWithGradKernel<<<1, batch_size, batch_size * sizeof(float), cuda_stream>>>(
    softmax_logits, log_softmax_logits, labels, batch_size, num_classes, loss, dx);
}

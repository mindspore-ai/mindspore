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
#include "sparse_cross_entropy_cuda_impl.cuh"

template <typename T>
__global__ void CalCrossEntropyKernel(const float *logits, T *labels, const int batch_size, const int class_num,
                                      float *loss) {
  float total_loss = 0.0;
  float epsilon = 1e-6;
  for (int i = 0; i < batch_size; ++i) {
    float logit = logits[i * class_num + labels[i]];
    if (logit <= 0) {
      logit += epsilon;
    }
    float single_loss = -logf(logit);
    total_loss += single_loss;
  }

  total_loss /= batch_size;
  loss[0] = total_loss;
  return;
}

template <typename T>
__global__ void CalCrossEntropyGradKernel(const float *logits, T *labels, const int batch_size, const int class_num,
                                          float *grad) {
  for (int i = 0; i < batch_size; i++) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < class_num; j += blockDim.x * gridDim.x) {
      if (labels[i] == j) {
        grad[i * class_num + j] = (logits[i * class_num + j] - 1) / batch_size;
      } else {
        grad[i * class_num + j] = logits[i * class_num + j] / batch_size;
      }
    }
  }
  return;
}

template <typename T>
cudaError_t CalCrossEntropy(const float *logits, T *labels, const int batch_size, const int class_num, float *loss,
                            cudaStream_t cuda_stream) {
  CalCrossEntropyKernel<<<1, 1, 0, cuda_stream>>>(logits, labels, batch_size, class_num, loss);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalCrossEntropyGrad(const float *logits, T *labels, const int batch_size, const int class_num, float *grad,
                                cudaStream_t cuda_stream) {
  CalCrossEntropyGradKernel<<<GET_BLOCKS(class_num), GET_THREADS, 0, cuda_stream>>>(logits, labels, batch_size,
                                                                                    class_num, grad);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalCrossEntropy<int>(const float *logits, int *labels, const int batch_size,
                                                          const int class_num, float *loss, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCrossEntropy<uint64_t>(const float *logits, uint64_t *labels,
                                                               const int batch_size, const int class_num, float *loss,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCrossEntropyGrad<int>(const float *logits, int *labels, const int batch_size,
                                                              const int class_num, float *grad,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCrossEntropyGrad<uint64_t>(const float *logits, uint64_t *labels,
                                                                   const int batch_size, const int class_num,
                                                                   float *grad, cudaStream_t cuda_stream);

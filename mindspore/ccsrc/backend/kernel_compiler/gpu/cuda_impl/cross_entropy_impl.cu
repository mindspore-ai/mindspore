/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "cross_entropy_impl.cuh"

template <typename T, typename S>
__global__ void CrossEntropyWithSparseKernel(const T *logits, const S *labels, const size_t batch_size,
                                             const size_t class_num, T *loss) {
  double total_loss = 0.0;
  T epsilon = 1e-6;
  for (size_t i = 0; i < batch_size; ++i) {
    T logit = logits[i * class_num + labels[i]];
    if (logit <= 0) {
      logit = epsilon;
    }
    total_loss += -logf(logit);
  }
  total_loss /= batch_size;
  loss[0] = static_cast<T>(total_loss);
  return;
}

template <typename T, typename S>
__global__ void CrossEntropyGradWithSparseKernel(const T *logits, const S *labels, const size_t batch_size,
                                                 const size_t class_num, T *grad) {
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < class_num; j += blockDim.x * gridDim.x) {
      if (labels[i] == j) {
        grad[i * class_num + j] = (logits[i * class_num + j] - 1) / batch_size;
      } else {
        grad[i * class_num + j] = logits[i * class_num + j] / batch_size;
      }
    }
  }
  return;
}

template <typename T, typename S>
__global__ void CrossEntropyKernel(const T *logits, const S *labels, const size_t class_num, T *losses, T *dlogits) {
  losses[threadIdx.x] = 0;
  T epsilon = 1e-6;
  for (int i = threadIdx.x * class_num; i < (threadIdx.x + 1) * class_num; ++i) {
    losses[threadIdx.x] -= logf((logits[i] <= 0 ? epsilon : logits[i])) * labels[i];
    dlogits[i] = logits[i] - labels[i];
  }
}

template <typename T, typename S>
void CrossEntropyWithSparse(const T *logits, const S *labels, const size_t batch_size, const size_t class_num, T *loss,
                            cudaStream_t cuda_stream) {
  CrossEntropyWithSparseKernel<<<1, 1, 0, cuda_stream>>>(logits, labels, batch_size, class_num, loss);
  return;
}

template <typename T, typename S>
void CrossEntropyGradWithSparse(const T *logits, const S *labels, const size_t batch_size, const size_t class_num,
                                T *grad, cudaStream_t cuda_stream) {
  CrossEntropyGradWithSparseKernel<<<GET_BLOCKS(class_num), GET_THREADS, 0, cuda_stream>>>(logits, labels, batch_size,
                                                                                           class_num, grad);
  return;
}

template <typename T, typename S>
void CrossEntropy(const T *logits, const S *labels, const size_t batch_size, const size_t class_num, T *losses,
                  T *dlogits, cudaStream_t cuda_stream) {
  CrossEntropyKernel<<<1, batch_size, 0, cuda_stream>>>(logits, labels, class_num, losses, dlogits);
}

template void CrossEntropyWithSparse<float, int>(const float *logits, const int *labels, const size_t batch_size,
                                                 const size_t class_num, float *loss, cudaStream_t cuda_stream);
template void CrossEntropyWithSparse<float, int64_t>(const float *logits, const int64_t *labels,
                                                     const size_t batch_size, const size_t class_num, float *loss,
                                                     cudaStream_t cuda_stream);
template void CrossEntropyGradWithSparse<float, int>(const float *logits, const int *labels, const size_t batch_size,
                                                     const size_t class_num, float *grad, cudaStream_t cuda_stream);
template void CrossEntropyGradWithSparse<float, int64_t>(const float *logits, const int64_t *labels,
                                                         const size_t batch_size, const size_t class_num, float *grad,
                                                         cudaStream_t cuda_stream);
template void CrossEntropy<float, float>(const float *logits, const float *labels, const size_t batch_size,
                                         const size_t class_num, float *losses, float *dlogits,
                                         cudaStream_t cuda_stream);

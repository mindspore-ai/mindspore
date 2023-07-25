/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

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
}

template <typename T, typename S>
__global__ void LargeBatchCrossEntropyWithSparseKernel(const T *logits, const S *labels, const size_t batch_size,
                                                       const size_t class_num, T *loss) {
  *loss = 0;
  T epsilon = 1e-6;
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size; index += blockDim.x * gridDim.x) {
    T logit = logits[index * class_num + labels[index]];
    if (logit <= 0) {
      logit = epsilon;
    }
    MsAtomicAdd(loss, -logf(logit) / batch_size);
  }
}

template <typename T, typename S>
__global__ void CrossEntropyGradWithSparseKernel(const T *logits, const S *labels, const size_t batch_size,
                                                 const size_t class_num, T *grad) {
  for (size_t i = 0; i < class_num; i++) {
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < batch_size; j += blockDim.x * gridDim.x) {
      if (labels[j] == i) {
        grad[j * class_num + i] = (logits[j * class_num + i] - 1) / batch_size;
      } else {
        grad[j * class_num + i] = logits[j * class_num + i] / batch_size;
      }
    }
  }
}

template <typename T, typename S>
__global__ void CrossEntropyKernel(const T *input0, const S *input1, const size_t batch_size, const size_t class_num,
                                   T *output0, T *output1, T *work) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size; index += blockDim.x * gridDim.x) {
    const float *logits = input0 + index * class_num;
    const float *labels = input1 + index * class_num;
    float *backprop = output1 + index * class_num;
    float *workspace = work + index * class_num;

    float maxv = logits[0];
    for (size_t i = 0; i < class_num; i++) {
      maxv = maxv > logits[i] ? maxv : logits[i];
    }

    float sum = 0.0;
    for (size_t i = 0; i < class_num; i++) {
      backprop[i] = logits[i] - maxv;
      workspace[i] = exp(backprop[i]);
      sum += workspace[i];
    }

    float logit = logf(sum);
    float loss = 0.0;

    for (size_t i = 0; i < class_num; i++) {
      loss += labels[i] * (backprop[i] - logit);
      workspace[i] = workspace[i] / sum;
      backprop[i] = workspace[i] - labels[i];
    }
    output0[index] = -loss;
  }
}

template <typename T, typename S>
cudaError_t CrossEntropyWithSparse(const T *logits, const S *labels, const size_t batch_size, const size_t class_num,
                                   T *loss, cudaStream_t cuda_stream) {
  if (batch_size <= kLargeBatchLowLimit) {
    CrossEntropyWithSparseKernel<<<1, 1, 0, cuda_stream>>>(logits, labels, batch_size, class_num, loss);
  } else {
    LargeBatchCrossEntropyWithSparseKernel<<<GET_BLOCKS(batch_size), GET_THREADS, 0, cuda_stream>>>(
      logits, labels, batch_size, class_num, loss);
  }
  return GetCudaStatus();
}

template <typename T, typename S>
cudaError_t CrossEntropyGradWithSparse(const T *logits, const S *labels, const size_t batch_size,
                                       const size_t class_num, T *grad, cudaStream_t cuda_stream) {
  CrossEntropyGradWithSparseKernel<<<GET_BLOCKS(batch_size), GET_THREADS, 0, cuda_stream>>>(logits, labels, batch_size,
                                                                                            class_num, grad);
  return GetCudaStatus();
}

template <typename T, typename S>
cudaError_t CrossEntropy(const T *logits, const S *labels, const size_t batch_size, const size_t class_num, T *losses,
                         T *dlogits, T *workspace, cudaStream_t cuda_stream) {
  CrossEntropyKernel<<<GET_BLOCKS(batch_size), GET_THREADS, 0, cuda_stream>>>(logits, labels, batch_size, class_num,
                                                                              losses, dlogits, workspace);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CrossEntropyWithSparse<float, int>(const float *logits, const int *labels,
                                                                        const size_t batch_size, const size_t class_num,
                                                                        float *loss, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CrossEntropyWithSparse<float, int64_t>(const float *logits, const int64_t *labels,
                                                                            const size_t batch_size,
                                                                            const size_t class_num, float *loss,
                                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CrossEntropyGradWithSparse<float, int>(const float *logits, const int *labels,
                                                                            const size_t batch_size,
                                                                            const size_t class_num, float *grad,
                                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CrossEntropyGradWithSparse<float, int64_t>(const float *logits,
                                                                                const int64_t *labels,
                                                                                const size_t batch_size,
                                                                                const size_t class_num, float *grad,
                                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CrossEntropy<float, float>(const float *logits, const float *labels,
                                                                const size_t batch_size, const size_t class_num,
                                                                float *losses, float *dlogits, float *workspace,
                                                                cudaStream_t cuda_stream);

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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/kernel_compiler/gpu/cuda_impl/sigmoid_cross_entropy_with_logits_grad_impl.cuh"

template <typename T, typename S>
__global__ void SigmoidCrossEntropyWithLogitsGradKernel(const size_t size, const T *logits, const S *labels,
                                                        const T *dout_addr, T *outputs) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (logits[i] >= 0) {
      outputs[i] = (static_cast<T>(1.) / (static_cast<T>(1.) + exp(-logits[i])) - labels[i]) * dout_addr[i];
    } else {
      const T exp_val = exp(logits[i]);
      outputs[i] = (exp_val / (static_cast<T>(1.) + exp_val) - labels[i]) * dout_addr[i];
    }
  }
}

template <typename T, typename S>
void SigmoidCrossEntropyWithLogitsGrad(const size_t size, const T *logits, const S *labels, const T *dout_addr,
                                       T *outputs, cudaStream_t cuda_stream) {
  SigmoidCrossEntropyWithLogitsGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, logits, labels,
                                                                                             dout_addr, outputs);
}

template void SigmoidCrossEntropyWithLogitsGrad<float, float>(const size_t size, const float *logits,
                                                              const float *labels, const float *dout_addr,
                                                              float *outputs, cudaStream_t cuda_stream);
template void SigmoidCrossEntropyWithLogitsGrad<double, double>(const size_t size, const double *logits,
                                                                const double *labels, const double *dout_addr,
                                                                double *outputs, cudaStream_t cuda_stream);

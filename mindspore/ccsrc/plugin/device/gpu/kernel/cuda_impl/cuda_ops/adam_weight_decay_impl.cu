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

#include "adam_weight_decay_impl.cuh"

template <typename T>
__global__ void AdamWeightDecayKernel(const int element_num_, const bool need_decay, const float *beta1,
                                      const float *one_sub_beta1, const float *beta2, const float *one_sub_beta2,
                                      const float *epsilon, const float *lr, const float *weight_decay, T *m, T *v,
                                      T *param, T *gradient) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < element_num_; i += blockDim.x * gridDim.x) {
    float next_m = beta1[0] * m[i] + one_sub_beta1[0] * gradient[i];
    float next_v = beta2[0] * v[i] + one_sub_beta2[0] * gradient[i] * gradient[i];
    float update = next_m / (sqrt(next_v) + epsilon[0]);
    if (need_decay && weight_decay != nullptr) {
      update += weight_decay[0] * param[i];
    }
    param[i] -= lr[0] * update;
    m[i] = next_m;
    v[i] = next_v;
  }
}

template <typename T>
cudaError_t AdamWeightDecay(const int &element_num_, const bool &need_decay, const float *beta1,
                            const float *one_sub_beta1, const float *beta2, const float *one_sub_beta2,
                            const float *epsilon, const float *lr, const float *weight_decay, T *m, T *v, T *param,
                            T *gradient, cudaStream_t stream) {
  AdamWeightDecayKernel<<<GET_BLOCKS(element_num_), GET_THREADS, 0, stream>>>(
    element_num_, need_decay, beta1, one_sub_beta1, beta2, one_sub_beta2, epsilon, lr, weight_decay, m, v, param,
    gradient);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t AdamWeightDecay(const int &element_num_, const bool &need_decay,
                                                     const float *beta1, const float *one_sub_beta1, const float *beta2,
                                                     const float *one_sub_beta2, const float *epsilon, const float *lr,
                                                     const float *weight_decay, float *m, float *v, float *param,
                                                     float *gradient, cudaStream_t stream);

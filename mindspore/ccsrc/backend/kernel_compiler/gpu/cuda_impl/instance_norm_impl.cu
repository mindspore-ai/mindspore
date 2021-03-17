/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/cuda_impl/instance_norm_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

__global__ void CopyMemKernel(const size_t thread_num, const size_t N, const size_t C,
                              float *gamma_addr, float *beta_addr,
                              float *runing_mean_addr, float *runnig_variance_addr,
                              float *ws_gamma, float *ws_beta, float *ws_mean, float *ws_var) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < thread_num; pos += gridDim.x * blockDim.x) {
    size_t cur_addr = pos / (N * C);
    size_t cur_local_index = pos % (N * C);
    size_t local_index = 0;
    switch (cur_addr) {
      case 0:
        if (!(gamma_addr && ws_gamma)) break;
        local_index = cur_local_index % C;
        ws_gamma[cur_local_index] = gamma_addr[local_index];
        break;
      case 1:
        if (!(beta_addr && ws_beta)) break;
        local_index = cur_local_index % C;
        ws_beta[cur_local_index] = beta_addr[local_index];
        break;
      case 2:
        if (!(runing_mean_addr && ws_mean)) break;
        local_index = cur_local_index % C;
        ws_mean[cur_local_index] = runing_mean_addr[local_index];
        break;
      default:
        if (!(runnig_variance_addr && ws_var)) break;
        local_index = cur_local_index % C;
        ws_var[cur_local_index] = runnig_variance_addr[local_index];
    }
  }
  return;
}

void CopyMemDevice2Device(const size_t N, const size_t C, float *gamma_addr, float *beta_addr,
                          float *runing_mean_addr, float *runnig_variance_addr,
                          float *ws_gamma, float *ws_beta, float *ws_mean, float *ws_var,
                          cudaStream_t cuda_stream) {
  size_t thread_num = N * C * 4;
  CopyMemKernel<<<GET_BLOCKS(thread_num), GET_THREADS, 0, cuda_stream>>>(
          thread_num, N, C, gamma_addr, beta_addr, runing_mean_addr, runnig_variance_addr,
          ws_gamma, ws_beta, ws_mean, ws_var);
}

__global__ void ComputeMeanKernel(const size_t thread_num, const size_t N, const size_t C,
                                  float *dgamma, float *dbeta, const float *ws_dgamma, const float *ws_dbeta) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < thread_num; pos += gridDim.x * blockDim.x) {
    size_t cur_addr = pos / C;
    size_t cur_local_index = pos % C;
    float tmp = 0;
    if (cur_addr) {
      for (size_t i = 0; i < N; i++) {
        tmp += ws_dgamma[i * C + cur_local_index];
      }
      dgamma[cur_local_index] = tmp;
    } else {
      for (size_t i = 0; i < N; i++) {
        tmp += ws_dbeta[i * C + cur_local_index];
      }
      dbeta[cur_local_index] = tmp;
    }
  }
  return;
}

void ComputeMean(const size_t N, const size_t C,
                 float *dgamma, float *dbeta, const float *ws_dgamma, const float *ws_dbeta,
                 cudaStream_t cuda_stream) {
  size_t thread_num = C * 2;
  ComputeMeanKernel<<<GET_BLOCKS(thread_num), GET_THREADS, 0, cuda_stream>>>(
          thread_num, N, C, dgamma, dbeta, ws_dgamma, ws_dbeta);
}

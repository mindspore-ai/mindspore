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

#include <algorithm>
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sync_batch_norm_impl.cuh"

const int kWarpSize = 32;
const int kNumWarps = 32;

__inline__ __device__ float HalfFloatInputConvert(const half val) { return __half2float(val); }
__inline__ __device__ float HalfFloatInputConvert(const float val) { return val; }
__inline__ __device__ void HalfFloatOutputAssign(const float val, float *arr, int idx) { arr[idx] = val; }
__inline__ __device__ void HalfFloatOutputAssign(const float val, half *arr, int idx) { arr[idx] = __float2half(val); }

template <typename T>
__global__ void SyncBatchNormPre(size_t N, size_t C, size_t H, size_t W, const T *input, int *output_n,
                                 float *output_mean, float *output_invstd, float epsilon) {
  // block level memory
  __shared__ float shared_mean[kNumWarps];
  __shared__ float shared_var[kNumWarps];
  __shared__ int shared_n[kNumWarps];

  int warpId = threadIdx.x / kWarpSize;  // threads execute in warps of 32
  int laneId = threadIdx.x % kWarpSize;
  int plane = blockIdx.x;
  int plane_size = N * H * W;
  if (threadIdx.x < kNumWarps) {
    shared_mean[threadIdx.x] = static_cast<float>(0);
    shared_var[threadIdx.x] = static_cast<float>(0);
  }
  // ensure all 0 init complete across all values
  __syncthreads();

  // agg values
  float avg = 0;
  float var_n = 0;
  int n = 0;

  // individual thread level reduction
  for (int x = threadIdx.x; x < plane_size; x += blockDim.x) {
    int index = (x / (H * W) * C * H * W) + (plane * H * W) + (x % (H * W));
    float input_val = HalfFloatInputConvert(input[index]);
    float d1 = input_val - avg;
    n++;
    avg = avg + (d1 / n);
    var_n = var_n + (d1 * (input_val - avg));
  }
  __syncthreads();

  // Reduce every warp to a single value
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    float other_avg = __shfl_down_sync(0xffffffff, avg, offset);
    float other_n = __shfl_down_sync(0xffffffff, n, offset);
    float div_factor = 1.0 / fmaxf(1.0, n + other_n);
    float other_var_n = __shfl_down_sync(0xffffffff, var_n, offset);
    var_n += other_var_n + (avg - other_avg) * (avg - other_avg) * n * other_n * div_factor;
    avg = (n * avg + other_n * other_avg) * div_factor;
    n += other_n;
  }
  __syncwarp();
  if (laneId == 0) {
    // lane 0 for every warp moves value
    shared_n[warpId] = n;
    shared_mean[warpId] = avg;
    shared_var[warpId] = var_n;
    // now one value per warp
  }
  // second reduction to reduce all warps into a single value
  __syncthreads();
  if (warpId == 0) {
    n = shared_n[laneId];
    avg = shared_mean[laneId];
    var_n = shared_var[laneId];
    __syncwarp();
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      int other_n = __shfl_down_sync(0xffffffff, n, offset);
      float other_avg = __shfl_down_sync(0xffffffff, avg, offset);
      float div_factor = 1.0 / fmaxf(1.0, n + other_n);
      float other_var_n = __shfl_down_sync(0xffffffff, var_n, offset);
      var_n += other_var_n + (avg - other_avg) * (avg - other_avg) * n * other_n * div_factor;
      avg = (n * avg + other_n * other_avg) * div_factor;
      n += other_n;
    }
    __syncwarp();
  }
  if (threadIdx.x == 0) {
    output_n[plane] = n;
    output_mean[plane] = avg;
    output_invstd[plane] = static_cast<float>(1) / sqrt((var_n / plane_size) + epsilon);
  }
  return;
}

template <typename T, typename G>
__global__ void SyncBatchNormGather(size_t N, size_t C, size_t H, size_t W, int *counts_global, float *means_global,
                                    float *invstds_global, int *counts_local, float *means_local, float *invstds_local,
                                    T *running_mean_output, T *running_var_output, G *running_mean_input,
                                    G *running_var_input, float epsilon, float momentum, size_t group_rank,
                                    size_t group_size) {
  int feature_size = C;
  int world_size = group_size;
  for (size_t C_ix = blockIdx.x * blockDim.x + threadIdx.x; C_ix < C; C_ix += blockDim.x * gridDim.x) {
    float avg = 0;
    float var_n = 0;
    float n = 0;
    for (int N_ix = 0; N_ix < world_size; N_ix++) {
      int count = counts_global[N_ix * feature_size + C_ix];
      float mean_ = means_global[N_ix * feature_size + C_ix];
      float std = static_cast<float>(1) / invstds_global[N_ix * feature_size + C_ix];
      float var_n_ = (std * std - epsilon) * count;
      float div_factor = 1.0 / fmaxf(1.0, n + count);
      var_n += var_n_ + (avg - mean_) * (avg - mean_) * n * count * div_factor;
      avg = n * div_factor * avg + count * div_factor * mean_;
      n += count;
    }
    means_local[C_ix] = avg;
    invstds_local[C_ix] = static_cast<float>(1) / sqrt((var_n / n) + epsilon);
    HalfFloatOutputAssign(((1 - momentum) * HalfFloatInputConvert(running_mean_input[C_ix]) + momentum * avg),
                          running_mean_output, C_ix);
    float unbiasedVar = 0.0;
    if (n != 0) {  // not strictly required since pipeline does not allow empty inputs
      unbiasedVar = var_n / n;
    }
    HalfFloatOutputAssign(((1 - momentum) * HalfFloatInputConvert(running_var_input[C_ix]) + momentum * unbiasedVar),
                          running_var_output, C_ix);
  }
  return;
}

template <typename T, typename S>
__global__ void SyncBatchNormPost(size_t N, size_t C, size_t H, size_t W, const T *input, T *output, float *means_local,
                                  float *invstds_local, S *scale, S *bias, float epsilon) {
  int size = N * C * H * W;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int block_num = (pos / W) / H;  // which of N * C blocks
    int plane = block_num % C;
    float scale_plane = HalfFloatInputConvert(scale[plane]);
    float bias_plane = HalfFloatInputConvert(bias[plane]);
    float mean_plane = means_local[plane];
    float invstd_plane = invstds_local[plane];
    float input_val = HalfFloatInputConvert(input[pos]);
    HalfFloatOutputAssign(scale_plane * (input_val - mean_plane) * invstd_plane + bias_plane, output, pos);
  }
  return;
}

template <typename S>
__global__ void SyncBatchNormPostBiasScale(size_t C, S *scale, S *bias, S *output_scale, S *output_bias) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < C; pos += blockDim.x * gridDim.x) {
    output_bias[pos] = bias[pos];
    output_scale[pos] = scale[pos];
  }
  return;
}

template <typename T>
void CalSyncBatchNormPre(size_t N, size_t C, size_t H, size_t W, const T *input, int *output_n, float *output_mean,
                         float *output_var, float epsilon, cudaStream_t cuda_stream) {
  SyncBatchNormPre<<<C, GET_THREADS, 0, cuda_stream>>>(N, C, H, W, input, output_n, output_mean, output_var, epsilon);
  return;
}

template <typename T, typename G>
void CalSyncBatchNormGather(size_t N, size_t C, size_t H, size_t W, int *counts_global, float *means_global,
                            float *invstds_global, int *counts_local, float *means_local, float *invstds_local,
                            T *running_mean_output, T *running_var_output, G *running_mean_input, G *running_var_input,
                            float epsilon, float momentum, size_t group_rank, size_t group_size,
                            cudaStream_t cuda_stream) {
  SyncBatchNormGather<<<GET_BLOCKS(C), GET_THREADS, 0, cuda_stream>>>(
    N, C, H, W, counts_global, means_global, invstds_global, counts_local, means_local, invstds_local,
    running_mean_output, running_var_output, running_mean_input, running_var_input, epsilon, momentum, group_rank,
    group_size);
  return;
}

template <typename T, typename S>
void CalSyncBatchNormPost(size_t N, size_t C, size_t H, size_t W, const T *input, T *output, float *means_local,
                          float *invstds_local, S *scale, S *bias, S *output_scale, S *output_bias, float epsilon,
                          cudaStream_t cuda_stream) {
  SyncBatchNormPost<<<GET_BLOCKS(N * C * H * W), GET_THREADS, 0, cuda_stream>>>(N, C, H, W, input, output, means_local,
                                                                                invstds_local, scale, bias, epsilon);
  SyncBatchNormPostBiasScale<<<1, std::min(C, static_cast<size_t>(GET_THREADS)), 0, cuda_stream>>>(
    C, scale, bias, output_scale, output_bias);
  return;
}

template void CalSyncBatchNormPre<float>(size_t N, size_t C, size_t H, size_t W, const float *input, int *output_n,
                                         float *output_mean, float *output_var, float epsilon,
                                         cudaStream_t cuda_stream);
template void CalSyncBatchNormPre<half>(size_t N, size_t C, size_t H, size_t W, const half *input, int *output_n,
                                        float *output_mean, float *output_var, float epsilon, cudaStream_t cuda_stream);

template void CalSyncBatchNormGather<float, float>(size_t N_, size_t C_, size_t H_, size_t W_, int *counts_global,
                                                   float *means_global, float *invstds_global, int *counts_local,
                                                   float *means_local, float *invstds_local, float *running_mean_output,
                                                   float *running_var_output, float *running_mean_input,
                                                   float *running_var_input, float epsilon, float momentum,
                                                   size_t group_rank, size_t group_size, cudaStream_t cuda_stream);
template void CalSyncBatchNormGather<float, half>(size_t N_, size_t C_, size_t H_, size_t W_, int *counts_global,
                                                  float *means_global, float *invstds_global, int *counts_local,
                                                  float *means_local, float *invstds_local, float *running_mean_output,
                                                  float *running_var_output, half *running_mean_input,
                                                  half *running_var_input, float epsilon, float momentum,
                                                  size_t group_rank, size_t group_size, cudaStream_t cuda_stream);
template void CalSyncBatchNormGather<half, float>(size_t N_, size_t C_, size_t H_, size_t W_, int *counts_global,
                                                  float *means_global, float *invstds_global, int *counts_local,
                                                  float *means_local, float *invstds_local, half *running_mean_output,
                                                  half *running_var_output, float *running_mean_input,
                                                  float *running_var_input, float epsilon, float momentum,
                                                  size_t group_rank, size_t group_size, cudaStream_t cuda_stream);
template void CalSyncBatchNormGather<half, half>(size_t N_, size_t C_, size_t H_, size_t W_, int *counts_global,
                                                 float *means_global, float *invstds_global, int *counts_local,
                                                 float *means_local, float *invstds_local, half *running_mean_output,
                                                 half *running_var_output, half *running_mean_input,
                                                 half *running_var_input, float epsilon, float momentum,
                                                 size_t group_rank, size_t group_size, cudaStream_t cuda_stream);

template void CalSyncBatchNormPost<float, float>(size_t N, size_t C, size_t H, size_t W, const float *input,
                                                 float *output, float *means_local, float *invstds_local, float *scale,
                                                 float *bias, float *output_scale, float *output_bias, float epsilon,
                                                 cudaStream_t cuda_stream);
template void CalSyncBatchNormPost<float, half>(size_t N, size_t C, size_t H, size_t W, const float *input,
                                                float *output, float *means_local, float *invstds_local, half *scale,
                                                half *bias, half *output_scale, half *output_bias, float epsilon,
                                                cudaStream_t cuda_stream);
template void CalSyncBatchNormPost<half, float>(size_t N, size_t C, size_t H, size_t W, const half *input, half *output,
                                                float *means_local, float *invstds_local, float *scale, float *bias,
                                                float *output_scale, float *output_bias, float epsilon,
                                                cudaStream_t cuda_stream);
template void CalSyncBatchNormPost<half, half>(size_t N, size_t C, size_t H, size_t W, const half *input, half *output,
                                               float *means_local, float *invstds_local, half *scale, half *bias,
                                               half *output_scale, half *output_bias, float epsilon,
                                               cudaStream_t cuda_stream);

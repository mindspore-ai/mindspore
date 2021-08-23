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
#include "backend/kernel_compiler/gpu/cuda_impl/sync_batch_norm_grad_impl.cuh"

const int kWarpSize = 32;
const int kNumWarps = 32;

__inline__ __device__ float HalfFloatInputConvert(const half val) { return __half2float(val); }
__inline__ __device__ float HalfFloatInputConvert(const float val) { return val; }
__inline__ __device__ void HalfFloatOutputAssign(const float val, float *arr, int idx) { arr[idx] = val; }
__inline__ __device__ void HalfFloatOutputAssign(const float val, half *arr, int idx) { arr[idx] = __float2half(val); }

template <typename T, typename G>
__global__ void SyncBatchNormGradPre(size_t N, size_t C, size_t H, size_t W, const T *x_input, const T *dy,
                                     G *saved_mean, G *saved_invstd, float *dy_sum_local, float *dot_p_local) {
  // block level memory
  __shared__ float shared_dy[kNumWarps];
  __shared__ float shared_dot_p[kNumWarps];
  int warpId = threadIdx.x / kWarpSize;  // threads are arranged in warps of 32 executed together
  int laneId = threadIdx.x % kWarpSize;

  int plane = blockIdx.x;  // this thread will only function on a single plane
  int plane_size = N * H * W;
  float mean = static_cast<float>(saved_mean[plane]);

  if (threadIdx.x < kNumWarps) {
    shared_dy[threadIdx.x] = static_cast<float>(0);
    shared_dot_p[threadIdx.x] = static_cast<float>(0);
  }

  __syncthreads();  // ensure all 0 init complete across all values

  float dy_sum = static_cast<float>(0);
  float dot_p = static_cast<float>(0);

  // individual thread level reduction
  for (int x = threadIdx.x; x < plane_size; x += blockDim.x) {
    int index = (x / (H * W) * C * H * W) + (plane * H * W) + (x % (H * W));
    float input_value = HalfFloatInputConvert(x_input[index]);
    float dy_value = HalfFloatInputConvert(dy[index]);
    dy_sum += dy_value;
    dot_p += (input_value - mean) * dy_value;
  }
  __syncthreads();
  // warp reduce all values in every value to a single value
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    float other_dy_sum = __shfl_down_sync(0xffffffff, dy_sum, offset);
    float other_dot_p = __shfl_down_sync(0xffffffff, dot_p, offset);
    dy_sum += other_dy_sum;
    dot_p += other_dot_p;
  }
  __syncwarp();
  if (laneId == 0) {
    shared_dy[warpId] = dy_sum;
    shared_dot_p[warpId] = dot_p;
    // one value per warp now
  }
  __syncthreads();
  if (warpId == 0) {
    dy_sum = shared_dy[laneId];
    dot_p = shared_dot_p[laneId];
    __syncwarp();
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      float other_dy = __shfl_down_sync(0xffffffff, dy_sum, offset);
      float other_dot_p = __shfl_down_sync(0xffffffff, dot_p, offset);
      dy_sum += other_dy;
      dot_p += other_dot_p;
    }
    __syncwarp();
  }
  if (threadIdx.x == 0) {
    dy_sum_local[plane] = dy_sum;
    dot_p_local[plane] = dot_p;
  }
  return;
}

template <typename T, typename S, typename G>
__global__ void SyncBatchNormGradPost(size_t N, size_t C, size_t H, size_t W, const T *x_input, const T *dy, T *dx,
                                      G *saved_mean, G *saved_invstd, float *dy_sum_red, float *dot_p_red, S *scale,
                                      S *dscale, S *dbias, float epsilon) {
  int size = N * C * H * W;
  int plane_size = N * H * W;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int block_num = (pos / W) / H;  // which of N * C blocks
    int plane = block_num % C;
    float mean = HalfFloatInputConvert(saved_mean[plane]);
    float invstd = HalfFloatInputConvert(saved_invstd[plane]);
    float scale_value = HalfFloatInputConvert(scale[plane]);
    float div_factor = HalfFloatInputConvert(1) / plane_size;
    float dy_sum_plane = dy_sum_red[plane];
    float dot_p_plane = dot_p_red[plane];
    float grad_mean = dy_sum_plane * div_factor;
    float proj_scale = dot_p_plane * div_factor * invstd * invstd;
    float grad_scale = invstd * scale_value;
    float inp = HalfFloatInputConvert(x_input[pos]);
    float proj = (inp - mean) * proj_scale;
    HalfFloatOutputAssign((HalfFloatInputConvert(dy[pos]) - proj - grad_mean) * grad_scale, dx, pos);
  }
}

template <typename S, typename G>
__global__ void SyncBatchNormGradPostScaleBias(size_t C, G *saved_invstd, float *dy_sum_red, float *dot_p_red,
                                               S *dscale, S *dbias) {
  for (size_t plane = blockIdx.x * blockDim.x + threadIdx.x; plane < C; plane += blockDim.x * gridDim.x) {
    float invstd = HalfFloatInputConvert(saved_invstd[plane]);
    float dy_sum_plane = dy_sum_red[plane];
    float dot_p_plane = dot_p_red[plane];
    dscale[plane] = static_cast<S>(dot_p_plane * invstd);
    dbias[plane] = static_cast<S>(dy_sum_plane);
  }
}

template <typename T, typename G>
void CalSyncBatchNormGradPre(size_t N, size_t C, size_t H, size_t W, const T *x_input, const T *dy, G *saved_mean,
                             G *saved_invstd, float *dy_sum_local, float *dot_p_local, cudaStream_t cuda_stream) {
  SyncBatchNormGradPre<<<C, GET_THREADS, 0, cuda_stream>>>(N, C, H, W, x_input, dy, saved_mean, saved_invstd,
                                                          dy_sum_local, dot_p_local);
  return;
}
template <typename T, typename S, typename G>
void CalSyncBatchNormGradPost(size_t N, size_t C, size_t H, size_t W, const T *x_input, const T *dy, T *dx,
                              G *saved_mean, G *saved_invstd, float *dy_sum_red, float *dot_p_red, S *scale, S *dscale,
                              S *dbias, float epsilon, cudaStream_t cuda_stream) {
  SyncBatchNormGradPost<<<C, GET_THREADS, 0, cuda_stream>>>(N, C, H, W, x_input, dy, dx, saved_mean, saved_invstd,
                                                            dy_sum_red, dot_p_red, scale, dscale, dbias, epsilon);
  SyncBatchNormGradPostScaleBias<<<GET_BLOCKS(C), std::min(C, static_cast<size_t>(GET_THREADS)), 0, cuda_stream>>>(
    C, saved_invstd, dy_sum_red, dot_p_red, dscale, dbias);
}
// PRE FUNCTION
template void CalSyncBatchNormGradPre<float, float>(size_t N, size_t C, size_t H, size_t W, const float *x_input,
                                                    const float *dy, float *saved_mean, float *saved_invstd,
                                                    float *dy_sum_local, float *dot_p_local, cudaStream_t cuda_stream);
template void CalSyncBatchNormGradPre<float, half>(size_t N, size_t C, size_t H, size_t W, const float *x_input,
                                                   const float *dy, half *saved_mean, half *saved_invstd,
                                                   float *dy_sum_local, float *dot_p_local, cudaStream_t cuda_stream);
template void CalSyncBatchNormGradPre<half, float>(size_t N, size_t C, size_t H, size_t W, const half *x_input,
                                                   const half *dy, float *saved_mean, float *saved_invstd,
                                                   float *dy_sum_local, float *dot_p_local, cudaStream_t cuda_stream);
template void CalSyncBatchNormGradPre<half, half>(size_t N, size_t C, size_t H, size_t W, const half *x_input,
                                                  const half *dy, half *saved_mean, half *saved_invstd,
                                                  float *dy_sum_local, float *dot_p_local, cudaStream_t cuda_stream);
// POST FUNCTION
template void CalSyncBatchNormGradPost<float, float, float>(size_t N, size_t C, size_t H, size_t W,
                                                            const float *x_input, const float *dy, float *dx,
                                                            float *saved_mean, float *saved_invstd, float *dy_sum_red,
                                                            float *dot_p_red, float *scale, float *dscale, float *dbias,
                                                            float epsilon, cudaStream_t cuda_stream);
template void CalSyncBatchNormGradPost<half, float, float>(size_t N, size_t C, size_t H, size_t W, const half *x_input,
                                                           const half *dy, half *dx, float *saved_mean,
                                                           float *saved_invstd, float *dy_sum_red, float *dot_p_red,
                                                           float *scale, float *dscale, float *dbias, float epsilon,
                                                           cudaStream_t cuda_stream);
template void CalSyncBatchNormGradPost<float, half, float>(size_t N, size_t C, size_t H, size_t W, const float *x_input,
                                                           const float *dy, float *dx, float *saved_mean,
                                                           float *saved_invstd, float *dy_sum_red, float *dot_p_red,
                                                           half *scale, half *dscale, half *dbias, float epsilon,
                                                           cudaStream_t cuda_stream);
template void CalSyncBatchNormGradPost<half, half, float>(size_t N, size_t C, size_t H, size_t W, const half *x_input,
                                                          const half *dy, half *dx, float *saved_mean,
                                                          float *saved_invstd, float *dy_sum_red, float *dot_p_red,
                                                          half *scale, half *dscale, half *dbias, float epsilon,
                                                          cudaStream_t cuda_stream);
template void CalSyncBatchNormGradPost<float, float, half>(size_t N, size_t C, size_t H, size_t W, const float *x_input,
                                                           const float *dy, float *dx, half *saved_mean,
                                                           half *saved_invstd, float *dy_sum_red, float *dot_p_red,
                                                           float *scale, float *dscale, float *dbias, float epsilon,
                                                           cudaStream_t cuda_stream);
template void CalSyncBatchNormGradPost<half, float, half>(size_t N, size_t C, size_t H, size_t W, const half *x_input,
                                                          const half *dy, half *dx, half *saved_mean,
                                                          half *saved_invstd, float *dy_sum_red, float *dot_p_red,
                                                          float *scale, float *dscale, float *dbias, float epsilon,
                                                          cudaStream_t cuda_stream);
template void CalSyncBatchNormGradPost<float, half, half>(size_t N, size_t C, size_t H, size_t W, const float *x_input,
                                                          const float *dy, float *dx, half *saved_mean,
                                                          half *saved_invstd, float *dy_sum_red, float *dot_p_red,
                                                          half *scale, half *dscale, half *dbias, float epsilon,
                                                          cudaStream_t cuda_stream);
template void CalSyncBatchNormGradPost<half, half, half>(size_t N, size_t C, size_t H, size_t W, const half *x_input,
                                                         const half *dy, half *dx, half *saved_mean, half *saved_invstd,
                                                         float *dy_sum_red, float *dot_p_red, half *scale, half *dscale,
                                                         half *dbias, float epsilon, cudaStream_t cuda_stream);

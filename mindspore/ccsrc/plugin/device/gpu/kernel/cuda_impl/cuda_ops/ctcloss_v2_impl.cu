/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/ctcloss_v2_impl.cuh"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <type_traits>
#include <limits>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

constexpr unsigned int kMaxThreads = 1024;
constexpr uint64_t kNumberWarps = 32;

template <typename T>
__device__ __forceinline__ T LogSumExp(T a, T b, T max_val) {
  return std::log(std::exp(a - max_val) + std::exp(b - max_val)) + max_val;
}

template <typename T>
__device__ __forceinline__ T LogSumExp(T a, T b, T c, T max_val) {
  return std::log(std::exp(a - max_val) + std::exp(b - max_val) + std::exp(c - max_val)) + max_val;
}

template <typename T>
__device__ __forceinline__ T DoLogSumExp(T a, T b) {
  constexpr T neg_inf = -std::numeric_limits<T>::infinity();
  if (a == neg_inf) {
    return b;
  } else {
    T max_val = max(a, b);
    return LogSumExp(a, b, max_val);
  }
}

struct LogSumExpFunc {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return DoLogSumExp(lhs, rhs);
  }
};

template <typename T>
__device__ __forceinline__ T AtomicLogSumExp(T *address, T val) {
  return atomic::MsAtomicBinaryOpImpl<LogSumExpFunc, T>()(address, val);
}

template <typename T>
__device__ __forceinline__ int64_t GetBlankPaddedTarget(const T *target, int64_t offset, int64_t idx, T blank) {
  constexpr int64_t interval = 2;
  if (idx % interval == 0) {
    return blank;
  } else {
    return target[offset + (idx / interval)];
  }
}

template <typename T>
__device__ __forceinline__ int64_t GetBlankPaddedTarget(const T *target, int64_t idx, T blank) {
  constexpr int64_t interval = 2;
  if (idx % interval == 0) {
    return blank;
  } else {
    return target[(idx / interval)];
  }
}

__device__ __forceinline__ size_t GetOffset3D(dim3 dims, size_t x, size_t y, size_t z) {
  return x * dims.y * dims.z + y * dims.z + z;
}

__device__ __forceinline__ void Revert3DIndex(dim3 dims, size_t index, size_t *x, size_t *y, size_t *z) {
  const size_t yz_offset = dims.y * dims.z;
  const size_t z_offset = dims.z;
  const size_t yz_index = index % yz_offset;
  *x = index / yz_offset;
  *y = yz_index / z_offset;
  *z = yz_index % z_offset;
}

template <typename T>
__global__ void CTCLossV2ShapeCheckKernel(const T *input_len_p, const T *target_len_p, int64_t max_target_length,
                                          int64_t time_series, int64_t batch_size) {
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < batch_size; b += blockDim.x * gridDim.x) {
    int64_t input_length = input_len_p[b];
    int64_t target_length = target_len_p[b];
    CUDA_KERNEL_ASSERT(input_length >= 0 && "For 'CTCLossV2', input_length should be non-negative.")
    CUDA_KERNEL_ASSERT(target_length >= 0 && "For 'CTCLossV2', target_length should be non-negative.")
    CUDA_KERNEL_ASSERT(target_length <= max_target_length &&
                       "For 'CTCLossV2', target_length should be less equal to targets.shape[1].")
    CUDA_KERNEL_ASSERT(input_length >= target_length &&
                       "For 'CTCLossV2', input_length should be greater equal to target_length.")
    CUDA_KERNEL_ASSERT(input_length <= time_series &&
                       "For 'CTCLossV2', input_length should be less equal to probs.shape[0].")
  }
}

template <typename S, typename T>
__global__ void CTCLossV2Kernel(const S *log_probs_p, const T *target_p, const T *input_len_p, const T *target_len_p,
                                int64_t max_target_length, int64_t batch_size, T blank, dim3 log_probs_shape,
                                dim3 log_alpha_shape, S *log_alpha_p) {
  int64_t batch = threadIdx.x + blockIdx.x * blockDim.x;
  if (batch >= batch_size) {
    return;
  }
  int64_t input_length = input_len_p[batch];
  int64_t target_length = target_len_p[batch];
  constexpr S neg_inf = -std::numeric_limits<S>::infinity();
  const int64_t offset = max_target_length * batch;
  const int64_t padded_max_target_length = 2 * max_target_length + 1;
  const int64_t padded_target_length = 2 * target_length + 1;
  // Init first line where t == 0
  for (int64_t block_s = 0; block_s < padded_max_target_length; block_s += blockDim.y) {
    int64_t s = block_s + threadIdx.y;
    if (s == 0) {
      log_alpha_p[GetOffset3D(log_alpha_shape, batch, 0, 0)] =
        log_probs_p[GetOffset3D(log_probs_shape, 0, batch, blank)];
    } else if (s == 1 && target_length > 0) {
      log_alpha_p[GetOffset3D(log_alpha_shape, batch, 0, 1)] =
        log_probs_p[GetOffset3D(log_probs_shape, 0, batch, GetBlankPaddedTarget(target_p, offset, 1, blank))];
    }
  }
  for (int64_t block_s = 0; block_s < padded_max_target_length; block_s += blockDim.y) {
    int64_t s = block_s + threadIdx.y;
    // Loop is based on max_target_length to
    if (s < padded_target_length) {
      bool valid_s = target_length > 0;
      auto current_target_prime = valid_s ? GetBlankPaddedTarget(target_p, offset, s, blank) : blank;
      bool three_sum =
        valid_s && (s > 1) && (GetBlankPaddedTarget(target_p, offset, s - 2, blank) != current_target_prime);
      // a1 is the result of the previous loop
      S log_a1 = log_alpha_p[GetOffset3D(log_alpha_shape, batch, 0, s)];
      // Starts with t = 1
      // Won't trigger warp divergence at the first time, even the number of trips in the for-loop is thread specific.
      // ref: https://forums.developer.nvidia.com/t/warp-divergence-triggered-by-for-loop/59769
      for (int64_t t = 1; t < input_length; t++) {
        __syncthreads();
        S log_max = log_a1;
        S log_a2, log_a3;
        if (s > 0) {
          log_a2 = log_alpha_p[GetOffset3D(log_alpha_shape, batch, t - 1, s - 1)];
          log_max = max(log_a2, log_max);
        } else {
          log_a2 = neg_inf;
        }
        if (three_sum) {
          log_a3 = log_alpha_p[GetOffset3D(log_alpha_shape, batch, t - 1, s - 2)];
          log_max = max(log_a3, log_max);
        } else {
          log_a3 = neg_inf;
        }
        if (log_max == neg_inf) {
          log_max = 0;
        }
        S log_three_sum = LogSumExp(log_a1, log_a2, log_a3, log_max) +
                          log_probs_p[GetOffset3D(log_probs_shape, t, batch, current_target_prime)];
        log_alpha_p[GetOffset3D(log_alpha_shape, batch, t, s)] = log_three_sum;
        log_a1 = log_three_sum;
      }
    }
  }
}

template <typename S, typename T>
__global__ void LogLikelihoodKernel(const S *log_alpha_p, const T *input_length_p, const T *target_length_p,
                                    int64_t batch_size, dim3 log_alpha_shape, S *neg_log_p) {
  constexpr S neg_inf = -std::numeric_limits<S>::infinity();
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < batch_size; b += blockDim.x * gridDim.x) {
    int64_t input_length = input_length_p[b];
    int64_t target_length = target_length_p[b];
    if (target_length == 0) {
      neg_log_p[b] = -log_alpha_p[GetOffset3D(log_alpha_shape, b, input_length - 1, 0)];
    } else {
      S l1 = log_alpha_p[GetOffset3D(log_alpha_shape, b, input_length - 1, target_length * 2)];
      S l2 = log_alpha_p[GetOffset3D(log_alpha_shape, b, input_length - 1, target_length * 2 - 1)];
      S max_val = max(l1, l2);
      max_val = ((max_val == neg_inf) ? 0 : max_val);
      S log_likelihood = LogSumExp(l1, l2, max_val);
      neg_log_p[b] = -log_likelihood;
    }
  }
}

template <typename S, typename T>
void CalCTCLossV2(const S *log_probs_p, const T *target_p, const T *input_len_p, const T *target_len_p,
                  int64_t batch_size, int64_t max_target_length, int64_t time_series, T blank, dim3 log_probs_shape,
                  dim3 log_alpha_shape, S *neg_log_p, S *log_alpha_p, uint32_t device_id, cudaStream_t cuda_stream) {
  constexpr S neg_inf = -std::numeric_limits<S>::infinity();
  const size_t alpha_size = log_alpha_shape.x * log_alpha_shape.y * log_alpha_shape.z;
  thrust::device_ptr<S> dev_ptr(log_alpha_p);
  thrust::fill(thrust::cuda::par.on(cuda_stream), dev_ptr, dev_ptr + alpha_size, neg_inf);

  const int64_t padded_target_length = 2 * max_target_length + 1;
  const uint64_t padded_target_length_power2 = 1ull << Log2Ceil64(padded_target_length);
  const uint64_t max_threads = CUDA_THREADS(device_id);
  const uint64_t threads_per_batch = std::min(max_threads, padded_target_length_power2);
  const unsigned int batches_per_block = std::min(max_threads / threads_per_batch, static_cast<uint64_t>(batch_size));

  dim3 blocks((batch_size + batches_per_block - 1) / batches_per_block);
  dim3 threads(batches_per_block, threads_per_batch);

  CTCLossV2ShapeCheckKernel<<<CUDA_BLOCKS(device_id, batch_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input_len_p, target_len_p, max_target_length, time_series, batch_size);

  CTCLossV2Kernel<<<blocks, threads, 0, cuda_stream>>>(log_probs_p, target_p, input_len_p, target_len_p,
                                                       max_target_length, batch_size, blank, log_probs_shape,
                                                       log_alpha_shape, log_alpha_p);
  LogLikelihoodKernel<<<CUDA_BLOCKS(device_id, batch_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    log_alpha_p, input_len_p, target_len_p, batch_size, log_alpha_shape, neg_log_p);
}

template <typename S, typename T>
__global__ void LogBetaKernel(const S *log_probs, const T *targets, const T *input_lengths, const T *target_lengths,
                              S *log_beta, int64_t batch_size, int64_t max_target_length, T blank, dim3 log_probs_shape,
                              dim3 log_beta_shape) {
  constexpr S neg_inf = -std::numeric_limits<S>::infinity();

  int64_t b = threadIdx.x + blockIdx.x * blockDim.x;
  if (b >= batch_size) {
    return;
  }

  const int64_t max_input_length = log_probs_shape.x;
  const int64_t offset = max_target_length * b;
  int64_t input_length = input_lengths[b];
  int64_t target_length = target_lengths[b];
  const int64_t padded_target_length = 2 * target_length + 1;

  // Init first line where t == 0
  for (int64_t block_s = 2 * max_target_length - (2 * max_target_length % blockDim.y); block_s >= 0;
       block_s -= blockDim.y) {
    int64_t s = block_s + threadIdx.y;
    if (s == 2 * target_length) {  // -1
      log_beta[GetOffset3D(log_beta_shape, b, input_length - 1, s)] =
        log_probs[GetOffset3D(log_probs_shape, input_length - 1, b, blank)];
    } else if ((s == 2 * target_length - 1) && (target_length > 0)) {  // -2
      auto current_target_prime = GetBlankPaddedTarget(targets, offset, s, blank);
      log_beta[GetOffset3D(log_beta_shape, b, input_length - 1, s)] =
        log_probs[GetOffset3D(log_probs_shape, input_length - 1, b, current_target_prime)];
    }
  }

  for (int64_t block_s = 2 * max_target_length - (2 * max_target_length % blockDim.y); block_s >= 0;
       block_s -= blockDim.y) {
    int64_t s = block_s + threadIdx.y;
    if (s < padded_target_length) {
      bool valid_s = target_length > 0;
      auto current_target_prime = valid_s ? GetBlankPaddedTarget(targets, offset, s, blank) : blank;
      bool three_sum = valid_s && (s < 2 * target_length - 1) &&
                       (GetBlankPaddedTarget(targets, offset, s + 2, blank) != current_target_prime);

      for (int64_t t = max_input_length - 2; t >= 0; t--) {
        __syncthreads();
        // t would start with different values in different threads if we loop over input_length directly.
        if (t < input_length - 1) {
          S log_b1 = log_beta[GetOffset3D(log_beta_shape, b, t + 1, s)];
          S log_max = log_b1;
          S log_b2, log_b3;
          if (s < 2 * target_length) {
            log_b2 = log_beta[GetOffset3D(log_beta_shape, b, t + 1, s + 1)];
            if (log_b2 > log_max) {
              log_max = log_b2;
            }
          } else {
            log_b2 = neg_inf;
          }
          if (three_sum) {
            log_b3 = log_beta[GetOffset3D(log_beta_shape, b, t + 1, s + 2)];
            if (log_b3 > log_max) {
              log_max = log_b3;
            }
          } else {
            log_b3 = neg_inf;
          }
          if (log_max == neg_inf) {
            log_max = 0;
          }
          log_beta[GetOffset3D(log_beta_shape, b, t, s)] =
            LogSumExp(log_b1, log_b2, log_b3, log_max) +
            log_probs[GetOffset3D(log_probs_shape, t, b, current_target_prime)];
        }
      }
    }
  }
}

template <typename S, typename T>
__global__ void AlphaBetaInitKernel(const T *targets, const T *input_lengths, const T *target_lengths,
                                    const S *neg_log_likelihood, const S *log_alpha, S *log_beta, int64_t batch_size,
                                    int64_t max_target_length, bool zero_infinity, T blank, dim3 log_probs_shape,
                                    dim3 log_alpha_shape, S *grad) {
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < batch_size; b += blockDim.x * gridDim.x) {
    S nll = neg_log_likelihood[b];
    if (zero_infinity && nll == std::numeric_limits<S>::infinity()) {
      continue;
    }
    int64_t input_length = input_lengths[b];
    int64_t target_length = target_lengths[b];
    int64_t offset = max_target_length * b;
    if (input_length > 0) {
      const auto alpha_beta_index = GetOffset3D(log_alpha_shape, b, input_length - 1, 2 * target_length);
      grad[GetOffset3D(log_probs_shape, input_length - 1, b, blank)] =
        log_alpha[alpha_beta_index] + log_beta[alpha_beta_index];
      if (target_length > 0) {
        const auto current_target_prime = GetBlankPaddedTarget(targets, offset, 2 * target_length - 1, blank);
        const auto alpha_beta_index = GetOffset3D(log_alpha_shape, b, input_length - 1, 2 * target_length - 1);
        grad[GetOffset3D(log_probs_shape, input_length - 1, b, current_target_prime)] =
          log_alpha[alpha_beta_index] + log_beta[alpha_beta_index];
      }
    }
  }
}

template <typename S, typename T>
__global__ void AlphaBetaComputeKernel(const T *targets, const T *input_lengths, const T *target_lengths,
                                       const S *neg_log_likelihood, const S *log_alpha, S *log_beta,
                                       int64_t alpha_beta_size, int64_t max_target_length, bool zero_infinity, T blank,
                                       dim3 alpha_beta_shape, dim3 log_probs_shape, dim3 log_alpha_shape, S *grad) {
  size_t t, b, s;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < alpha_beta_size;
       index += blockDim.x * gridDim.x) {
    Revert3DIndex(alpha_beta_shape, index, &t, &b, &s);
    int64_t input_length = input_lengths[b];
    if (t >= input_length - 1) {
      continue;
    }
    S nll = neg_log_likelihood[b];
    if (zero_infinity && nll == std::numeric_limits<S>::infinity()) {
      continue;
    }
    const int64_t target_length = target_lengths[b];
    if (s >= target_length) {
      continue;
    }

    const size_t padded_s = s * 2 + 1;
    const size_t offset = max_target_length * b;
    const auto current_target_prime = GetBlankPaddedTarget<T>(targets, offset, padded_s, blank);
    const auto alpha_beta_index = GetOffset3D(log_alpha_shape, b, t, padded_s);
    S log_alpha_beta = log_alpha[alpha_beta_index] + log_beta[alpha_beta_index];
    AtomicLogSumExp(&grad[GetOffset3D(log_probs_shape, t, b, current_target_prime)], log_alpha_beta);
  }
}

template <typename S, typename T>
__global__ void AlphaBetaBlankKernel(const T *input_lengths, const T *target_lengths, const S *neg_log_likelihood,
                                     const S *log_alpha, S *log_beta, int64_t alpha_beta_size, int64_t stride,
                                     bool zero_infinity, T blank, dim3 log_probs_shape, dim3 log_alpha_shape, S *grad) {
  __shared__ S workspace[kMaxThreads];
  // Calculate current workspace
  const unsigned int tid = threadIdx.x;

  constexpr S neg_inf = -std::numeric_limits<S>::infinity();
  int64_t index = blockIdx.x;
  int64_t b = index / stride;
  int64_t t = index % stride;
  int64_t input_length = input_lengths[b];
  S nll = neg_log_likelihood[b];
  int64_t target_length = target_lengths[b];
  // check
  if ((index >= alpha_beta_size) || (t >= input_length - 1) ||
      (zero_infinity && nll == std::numeric_limits<S>::infinity()) || (tid >= (target_length + 1))) {
    workspace[tid] = neg_inf;
    return;
  }

  // Init workspace
  auto alpha_beta_index = GetOffset3D(log_alpha_shape, b, t, tid * 2);
  workspace[tid] = log_alpha[alpha_beta_index] + log_beta[alpha_beta_index];

  for (int64_t s = (tid + blockDim.x) * 2; s < 2 * target_length + 1; s += blockDim.x * 2) {
    alpha_beta_index = GetOffset3D(log_alpha_shape, b, t, s);
    S log_alpha_beta = log_alpha[alpha_beta_index] + log_beta[alpha_beta_index];
    workspace[tid] = DoLogSumExp(workspace[tid], log_alpha_beta);
  }

  // Reduce inside shared memory
  for (unsigned int reduce_size = kMaxThreads; reduce_size >= 128; reduce_size /= 2) {
    __syncthreads();
    unsigned int half_reduce = reduce_size / 2;
    if (blockDim.x >= reduce_size && tid < half_reduce) {
      workspace[tid] = DoLogSumExp(workspace[tid], workspace[tid + half_reduce]);
    }
  }

  __syncthreads();
  if (tid < 32) {
    volatile S *warp_workspace = workspace;
    warp_workspace[tid] = DoLogSumExp(warp_workspace[tid], warp_workspace[tid + 32]);
    warp_workspace[tid] = DoLogSumExp(warp_workspace[tid], warp_workspace[tid + 16]);
    warp_workspace[tid] = DoLogSumExp(warp_workspace[tid], warp_workspace[tid + 8]);
    warp_workspace[tid] = DoLogSumExp(warp_workspace[tid], warp_workspace[tid + 4]);
    warp_workspace[tid] = DoLogSumExp(warp_workspace[tid], warp_workspace[tid + 2]);
    warp_workspace[tid] = DoLogSumExp(warp_workspace[tid], warp_workspace[tid + 1]);
  }

  if (tid == 0) {
    grad[GetOffset3D(log_probs_shape, t, b, blank)] = workspace[0];
  }
}

template <typename S, typename T>
__global__ void GradOutKernel(const S *grad_out, const S *log_probs, const T *input_lengths,
                              const S *neg_log_likelihood, bool zero_infinity, dim3 log_probs_shape, S *grad) {
  const auto grad_size = log_probs_shape.x * log_probs_shape.y * log_probs_shape.z;
  size_t t, b, c;
  // 3D index with shape time * batch * targets
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < grad_size; index += blockDim.x * gridDim.x) {
    Revert3DIndex(log_probs_shape, index, &t, &b, &c);
    int64_t input_length = input_lengths[b];
    S nll = neg_log_likelihood[b];
    if (zero_infinity && nll == std::numeric_limits<S>::infinity()) {
      grad[GetOffset3D(log_probs_shape, t, b, c)] = 0;
    } else {
      S gr = grad_out[b];
      if (t < input_length) {
        S &res = grad[GetOffset3D(log_probs_shape, t, b, c)];
        S lp = log_probs[GetOffset3D(log_probs_shape, t, b, c)];
        res = (std::exp(lp) - std::exp(res + nll - lp)) * gr;
      } else {
        grad[GetOffset3D(log_probs_shape, t, b, c)] = 0;
      }
    }
  }
}

template <typename S, typename T>
void CalCTCLossGradV2(const S *grad_out, const S *log_probs, const T *targets, const T *input_lengths,
                      const T *target_lengths, const S *neg_log_likelihood, const S *log_alpha, S *log_beta,
                      int64_t batch_size, int64_t time_series, int64_t max_target_length, bool zero_infinity, T blank,
                      dim3 log_probs_shape, dim3 log_alpha_shape, S *grad, uint32_t device_id,
                      cudaStream_t cuda_stream) {
  constexpr S neg_inf = -std::numeric_limits<S>::infinity();

  const size_t grad_size = log_probs_shape.x * log_probs_shape.y * log_probs_shape.z;
  thrust::device_ptr<S> dev_ptr(grad);
  thrust::fill(thrust::cuda::par.on(cuda_stream), dev_ptr, dev_ptr + grad_size, neg_inf);

  const size_t beta_size = log_alpha_shape.x * log_alpha_shape.y * log_alpha_shape.z;
  thrust::device_ptr<S> beta_dev_ptr(log_beta);
  thrust::fill(thrust::cuda::par.on(cuda_stream), beta_dev_ptr, beta_dev_ptr + beta_size, neg_inf);

  const uint64_t max_threads = CUDA_THREADS(device_id);
  const int64_t padded_target_length = 2 * max_target_length + 1;
  const uint64_t padded_target_length_power2 = 1ull << Log2Ceil64(padded_target_length);
  const uint64_t threads_per_batch = std::min(max_threads, padded_target_length_power2);
  const unsigned int batches_per_block = std::min(max_threads / threads_per_batch, static_cast<uint64_t>(batch_size));

  dim3 blocks((batch_size + batches_per_block - 1) / batches_per_block);
  dim3 threads(batches_per_block, threads_per_batch);

  LogBetaKernel<<<blocks, threads, 0, cuda_stream>>>(log_probs, targets, input_lengths, target_lengths, log_beta,
                                                     batch_size, max_target_length, blank, log_probs_shape,
                                                     log_alpha_shape);

  AlphaBetaInitKernel<<<CUDA_BLOCKS(device_id, batch_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, log_beta, batch_size, max_target_length,
    zero_infinity, blank, log_probs_shape, log_alpha_shape, grad);

  dim3 alpha_beta_shape(time_series - 1, batch_size, max_target_length);
  const size_t alpha_beta_size = batch_size * (time_series - 1) * max_target_length;

  AlphaBetaComputeKernel<<<CUDA_BLOCKS(device_id, alpha_beta_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, log_beta, alpha_beta_size, max_target_length,
    zero_infinity, blank, alpha_beta_shape, log_probs_shape, log_alpha_shape, grad);

  const size_t time_batch_size = batch_size * (time_series - 1);
  const int64_t blank_target_length = max_target_length + 1;
  const uint64_t target_length_power2 = 1ull << Log2Ceil64(blank_target_length);
  const uint64_t threads_per_block = std::max(std::min(max_threads, target_length_power2), kNumberWarps * 2);

  AlphaBetaBlankKernel<S, T><<<time_batch_size, threads_per_block, 0, cuda_stream>>>(
    input_lengths, target_lengths, neg_log_likelihood, log_alpha, log_beta, time_batch_size, (time_series - 1),
    zero_infinity, blank, log_probs_shape, log_alpha_shape, grad);

  GradOutKernel<<<CUDA_BLOCKS(device_id, grad_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    grad_out, log_probs, input_lengths, neg_log_likelihood, zero_infinity, log_probs_shape, grad);
}

template CUDA_LIB_EXPORT void CalCTCLossV2<float, int>(const float *log_probs_p, const int *target_p,
                                                       const int *input_len_p, const int *target_len_p,
                                                       int64_t batch_size, int64_t target_stride, int64_t time_series,
                                                       int blank, dim3 log_probs_shape, dim3 log_alpha_shape,
                                                       float *neg_log_p, float *log_alpha_p, uint32_t device_id,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalCTCLossV2<double, int>(const double *log_probs_p, const int *target_p,
                                                        const int *input_len_p, const int *target_len_p,
                                                        int64_t batch_size, int64_t target_stride, int64_t time_series,
                                                        int blank, dim3 log_probs_shape, dim3 log_alpha_shape,
                                                        double *neg_log_p, double *log_alpha_p, uint32_t device_id,
                                                        cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCTCLossV2<float, int64_t>(const float *log_probs_p, const int64_t *target_p,
                                                           const int64_t *input_len_p, const int64_t *target_len_p,
                                                           int64_t batch_size, int64_t target_stride,
                                                           int64_t time_series, int64_t blank, dim3 log_probs_shape,
                                                           dim3 log_alpha_shape, float *neg_log_p, float *log_alpha_p,
                                                           uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCTCLossV2<double, int64_t>(
  const double *log_probs_p, const int64_t *target_p, const int64_t *input_len_p, const int64_t *target_len_p,
  int64_t batch_size, int64_t target_stride, int64_t time_series, int64_t blank, dim3 log_probs_shape,
  dim3 log_alpha_shape, double *neg_log_p, double *log_alpha_p, uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCTCLossGradV2<float, int>(
  const float *grad_out, const float *log_probs, const int *targets, const int *input_lengths,
  const int *target_lengths, const float *neg_log_likelihood, const float *log_alpha, float *log_beta,
  int64_t batch_size, int64_t time_series, int64_t max_target_length, bool zero_infinity, int blank,
  dim3 log_probs_shape, dim3 log_alpha_shape, float *grad, uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCTCLossGradV2<double, int>(
  const double *grad_out, const double *log_probs, const int *targets, const int *input_lengths,
  const int *target_lengths, const double *neg_log_likelihood, const double *log_alpha, double *log_beta,
  int64_t batch_size, int64_t time_series, int64_t max_target_length, bool zero_infinity, int blank,
  dim3 log_probs_shape, dim3 log_alpha_shape, double *grad, uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCTCLossGradV2<float, int64_t>(
  const float *grad_out, const float *log_probs, const int64_t *targets, const int64_t *input_lengths,
  const int64_t *target_lengths, const float *neg_log_likelihood, const float *log_alpha, float *log_beta,
  int64_t batch_size, int64_t time_series, int64_t max_target_length, bool zero_infinity, int64_t blank,
  dim3 log_probs_shape, dim3 log_alpha_shape, float *grad, uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCTCLossGradV2<double, int64_t>(
  const double *grad_out, const double *log_probs, const int64_t *targets, const int64_t *input_lengths,
  const int64_t *target_lengths, const double *neg_log_likelihood, const double *log_alpha, double *log_beta,
  int64_t batch_size, int64_t time_series, int64_t max_target_length, bool zero_infinity, int64_t blank,
  dim3 log_probs_shape, dim3 log_alpha_shape, double *grad, uint32_t device_id, cudaStream_t cuda_stream);

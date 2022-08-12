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
#include <limits>
#include <algorithm>

template <typename T>
__device__ __forceinline__ T LogSumExp(T a, T b) {
  constexpr T neg_inf = -std::numeric_limits<T>::infinity();
  if (a < b) {
    T tmp = a;
    a = b;
    b = tmp;
  }
  if (b == neg_inf) {
    return a;
  }
  return a + log1p(exp(b - a));
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

__device__ __forceinline__ size_t GetOffset3D(dim3 dims, size_t x, size_t y, size_t z) {
  return x * dims.y * dims.z + y * dims.z + z;
}

template <typename S, typename T>
__device__ __forceinline__ void LossCompute(const S *log_probs_p, S *log_alpha_p, const T *tar_p, int64_t input_length,
                                            int64_t target_length, int64_t offset, int64_t batch, T blank,
                                            dim3 log_probs_shape, dim3 log_alpha_shape) {
  constexpr S neg_inf = -std::numeric_limits<S>::infinity();
  if (target_length > 0) {
    log_alpha_p[GetOffset3D(log_alpha_shape, batch, 0, 1)] =
      log_probs_p[GetOffset3D(log_probs_shape, 0, batch, GetBlankPaddedTarget(tar_p, offset, 1, blank))];
  }
  for (int64_t t = 1; t < input_length; t++) {
    for (int64_t s = 0; s < 2 * target_length + 1; s++) {
      auto current_target_prime = GetBlankPaddedTarget(tar_p, offset, s, blank);
      S log_a1 = log_alpha_p[GetOffset3D(log_alpha_shape, batch, t - 1, s)];
      S log_max = log_a1;
      S log_a2, log_a3;
      if (s > 0) {
        log_a2 = log_alpha_p[GetOffset3D(log_alpha_shape, batch, t - 1, s - 1)];
        log_max = max(log_a2, log_max);
      } else {
        log_a2 = neg_inf;
      }
      if ((s > 1) && (GetBlankPaddedTarget(tar_p, offset, s - 2, blank) != current_target_prime)) {
        log_a3 = log_alpha_p[GetOffset3D(log_alpha_shape, batch, t - 1, s - 2)];
        log_max = max(log_a3, log_max);
      } else {
        log_a3 = neg_inf;
      }
      if (log_max == neg_inf) {
        log_max = 0;
      }
      log_alpha_p[GetOffset3D(log_alpha_shape, batch, t, s)] =
        std::log(std::exp(log_a1 - log_max) + std::exp(log_a2 - log_max) + std::exp(log_a3 - log_max)) + log_max +
        log_probs_p[GetOffset3D(log_probs_shape, t, batch, current_target_prime)];
    }
  }
}

template <typename S, typename T>
__device__ __forceinline__ void GradCompute(const S *log_probs, const S *log_alpha, S *log_beta, int64_t blank,
                                            int64_t input_length, int64_t target_length, int64_t tg_batch_offset,
                                            int64_t b, const T *targets, dim3 log_probs_shape, dim3 log_alpha_shape,
                                            S *grad) {
  constexpr S neg_inf = -std::numeric_limits<S>::infinity();
  for (int64_t t = input_length - 2; t >= 0; t--) {
    for (int64_t s = 2 * target_length; s >= 0; s--) {
      S lb1 = log_beta[GetOffset3D(log_alpha_shape, b, t + 1, s)];
      S lbmax = lb1;
      S lb2, lb3;
      auto current_target_prime = GetBlankPaddedTarget<T>(targets, tg_batch_offset, s, blank);
      if (s < 2 * target_length) {
        lb2 = log_beta[GetOffset3D(log_alpha_shape, b, t + 1, s + 1)];
        if (lb2 > lbmax) {
          lbmax = lb2;
        }
      } else {
        lb2 = neg_inf;
      }
      if ((s < 2 * target_length - 1) &&
          (GetBlankPaddedTarget<T>(targets, tg_batch_offset, s + 2, blank) != current_target_prime)) {
        lb3 = log_beta[GetOffset3D(log_alpha_shape, b, t + 1, s + 2)];
        if (lb3 > lbmax) {
          lbmax = lb3;
        }
      } else {
        lb3 = neg_inf;
      }
      if (lbmax == neg_inf) {
        lbmax = 0;
      }
      log_beta[GetOffset3D(log_alpha_shape, b, t, s)] =
        std::log(std::exp(lb1 - lbmax) + std::exp(lb2 - lbmax) + std::exp(lb3 - lbmax)) + lbmax +
        log_probs[GetOffset3D(log_probs_shape, t, b, current_target_prime)];
      S log_alpha_beta =
        log_alpha[GetOffset3D(log_alpha_shape, b, t, s)] + log_beta[GetOffset3D(log_alpha_shape, b, t, s)];
      S &lcab = grad[GetOffset3D(log_probs_shape, t, b, current_target_prime)];
      if (lcab == neg_inf) {
        lcab = log_alpha_beta;
      } else {
        S max_val = max(lcab, log_alpha_beta);
        lcab = std::log(std::exp(lcab - max_val) + std::exp(log_alpha_beta - max_val)) + max_val;
      }
    }
  }
}

template <typename S, typename T>
__global__ void CTCLossV2Kernel(const S *log_probs_p, const T *target_p, const T *input_len_p, const T *target_len_p,
                                int64_t max_target_length, int64_t time_series, int64_t batch_size, T blank,
                                dim3 log_probs_shape, dim3 log_alpha_shape, S *neg_log_p, S *log_alpha_p) {
  constexpr S neg_inf = -std::numeric_limits<S>::infinity();
  for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < batch_size; b += blockDim.x * gridDim.x) {
    int64_t input_len = input_len_p[b];
    int64_t tar_len = target_len_p[b];
    CUDA_KERNEL_ASSERT(input_len >= 0 && "For 'CTCLossV2', input_length should be non-negative.")
    CUDA_KERNEL_ASSERT(tar_len >= 0 && "For 'CTCLossV2', target_length should be non-negative.")
    CUDA_KERNEL_ASSERT(tar_len <= max_target_length &&
                       "For 'CTCLossV2', target_length should be less equal to targets.shape[1].")
    CUDA_KERNEL_ASSERT(input_len >= tar_len &&
                       "For 'CTCLossV2', input_length should be greater equal to target_length.")
    CUDA_KERNEL_ASSERT(input_len <= time_series &&
                       "For 'CTCLossV2', input_length should be less equal to probs.shape[0].")
    int64_t offset = max_target_length * b;
    log_alpha_p[GetOffset3D(log_alpha_shape, b, 0, 0)] = log_probs_p[GetOffset3D(log_probs_shape, 0, b, blank)];

    LossCompute<S, T>(log_probs_p, log_alpha_p, target_p, input_len, tar_len, offset, b, blank, log_probs_shape,
                      log_alpha_shape);
    if (tar_len == 0) {
      neg_log_p[b] = -log_alpha_p[GetOffset3D(log_alpha_shape, b, input_len - 1, 0)];
    } else {
      S l1 = log_alpha_p[GetOffset3D(log_alpha_shape, b, input_len - 1, tar_len * 2)];
      S l2 = log_alpha_p[GetOffset3D(log_alpha_shape, b, input_len - 1, tar_len * 2 - 1)];
      S m = max(l1, l2);
      m = ((m == neg_inf) ? 0 : m);
      S log_likelihood = std::log(std::exp(l1 - m) + std::exp(l2 - m)) + m;
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

  CTCLossV2Kernel<<<CUDA_BLOCKS(device_id, batch_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    log_probs_p, target_p, input_len_p, target_len_p, max_target_length, time_series, batch_size, blank,
    log_probs_shape, log_alpha_shape, neg_log_p, log_alpha_p);
}

template <typename S, typename T>
__global__ void CTCLossV2GradKernel(const S *grad_out, const S *log_probs, const T *targets, const T *input_lengths,
                                    const T *target_lengths, const S *neg_log_likelihood, const S *log_alpha,
                                    S *log_beta, int64_t batch_size, int64_t time_series, int64_t num_labels,
                                    int64_t max_target_length, bool zero_infinity, T blank, dim3 log_probs_shape,
                                    dim3 log_alpha_shape, S *grad) {
  constexpr S neg_inf = -std::numeric_limits<S>::infinity();
  for (int64_t b = 0; b < batch_size; b++) {
    S nll = neg_log_likelihood[b];
    if (zero_infinity && nll == std::numeric_limits<S>::infinity()) {
      for (int t = 0; t < time_series; t++) {
        for (int c = 0; c < num_labels; c++) {
          grad[GetOffset3D(log_probs_shape, t, b, c)] = 0;
        }
      }
      continue;
    }
    int64_t input_length = input_lengths[b];
    int64_t target_length = target_lengths[b];
    int64_t tg_batch_offset = max_target_length * b;
    if (input_length > 0) {
      for (size_t s = 0; s < 2 * max_target_length + 1; s++) {
        log_beta[GetOffset3D(log_alpha_shape, b, input_length - 1, s)] = neg_inf;
      }
      log_beta[GetOffset3D(log_alpha_shape, b, input_length - 1, 2 * target_length)] =
        log_probs[GetOffset3D(log_probs_shape, input_length - 1, b, blank)];
      grad[GetOffset3D(log_probs_shape, input_length - 1, b, blank)] =
        log_alpha[GetOffset3D(log_alpha_shape, b, input_length - 1, 2 * target_length)] +
        log_beta[GetOffset3D(log_alpha_shape, b, input_length - 1, 2 * target_length)];
      if (target_length > 0) {
        auto current_target_prime = GetBlankPaddedTarget(targets, tg_batch_offset, 2 * target_length - 1, blank);
        log_beta[GetOffset3D(log_alpha_shape, b, input_length - 1, 2 * target_length - 1)] =
          log_probs[GetOffset3D(log_probs_shape, input_length - 1, b, current_target_prime)];
        grad[GetOffset3D(log_probs_shape, input_length - 1, b, current_target_prime)] =
          log_alpha[GetOffset3D(log_alpha_shape, b, input_length - 1, 2 * target_length - 1)] +
          log_beta[GetOffset3D(log_alpha_shape, b, input_length - 1, 2 * target_length - 1)];
      }
    }
    GradCompute<S, T>(log_probs, log_alpha, log_beta, blank, input_length, target_length, tg_batch_offset, b, targets,
                      log_probs_shape, log_alpha_shape, grad);
    S gr = grad_out[b];
    for (int64_t t = 0; t < input_length; t++) {
      for (int64_t c = 0; c < num_labels; c++) {
        S &res = grad[GetOffset3D(log_probs_shape, t, b, c)];
        S lp = log_probs[GetOffset3D(log_probs_shape, t, b, c)];
        res = (std::exp(lp) - std::exp(res + nll - lp)) * gr;
      }
    }
    for (auto l = input_length; l < time_series; l++) {
      for (int c = 0; c < num_labels; c++) {
        grad[GetOffset3D(log_probs_shape, l, b, c)] = 0;
      }
    }
  }
}

template <typename S, typename T>
void CalCTCLossGradV2(const S *grad_out, const S *log_probs, const T *targets, const T *input_lengths,
                      const T *target_lengths, const S *neg_log_likelihood, const S *log_alpha, S *log_beta,
                      int64_t batch_size, int64_t time_series, int64_t num_labels, int64_t max_target_length,
                      bool zero_infinity, T blank, dim3 log_probs_shape, dim3 log_alpha_shape, S *grad,
                      uint32_t device_id, cudaStream_t cuda_stream) {
  constexpr S neg_inf = -std::numeric_limits<S>::infinity();
  const size_t grad_size = log_probs_shape.x * log_probs_shape.y * log_probs_shape.z;
  thrust::device_ptr<S> dev_ptr(grad);
  thrust::fill(thrust::cuda::par.on(cuda_stream), dev_ptr, dev_ptr + grad_size, neg_inf);

  CTCLossV2GradKernel<<<CUDA_BLOCKS(device_id, batch_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    grad_out, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, log_beta, batch_size,
    time_series, num_labels, max_target_length, zero_infinity, blank, log_probs_shape, log_alpha_shape, grad);
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
  int64_t batch_size, int64_t time_series, int64_t num_labels, int64_t max_target_length, bool zero_infinity, int blank,
  dim3 log_probs_shape, dim3 log_alpha_shape, float *grad, uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCTCLossGradV2<double, int>(
  const double *grad_out, const double *log_probs, const int *targets, const int *input_lengths,
  const int *target_lengths, const double *neg_log_likelihood, const double *log_alpha, double *log_beta,
  int64_t batch_size, int64_t time_series, int64_t num_labels, int64_t max_target_length, bool zero_infinity, int blank,
  dim3 log_probs_shape, dim3 log_alpha_shape, double *grad, uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCTCLossGradV2<float, int64_t>(
  const float *grad_out, const float *log_probs, const int64_t *targets, const int64_t *input_lengths,
  const int64_t *target_lengths, const float *neg_log_likelihood, const float *log_alpha, float *log_beta,
  int64_t batch_size, int64_t time_series, int64_t num_labels, int64_t max_target_length, bool zero_infinity,
  int64_t blank, dim3 log_probs_shape, dim3 log_alpha_shape, float *grad, uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCTCLossGradV2<double, int64_t>(
  const double *grad_out, const double *log_probs, const int64_t *targets, const int64_t *input_lengths,
  const int64_t *target_lengths, const double *neg_log_likelihood, const double *log_alpha, double *log_beta,
  int64_t batch_size, int64_t time_series, int64_t num_labels, int64_t max_target_length, bool zero_infinity,
  int64_t blank, dim3 log_probs_shape, dim3 log_alpha_shape, double *grad, uint32_t device_id,
  cudaStream_t cuda_stream);

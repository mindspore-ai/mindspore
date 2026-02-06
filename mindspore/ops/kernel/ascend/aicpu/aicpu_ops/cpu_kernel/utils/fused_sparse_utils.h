/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifndef AICPU_UTILS_FUSED_SPARSE_UTIL_H_
#define AICPU_UTILS_FUSED_SPARSE_UTIL_H_

#include "inc/cpu_context.h"
#include "include/securec.h"
#if __has_include("context/common/cpu_kernel_utils.h")
#include "context/common/cpu_kernel_utils.h"
#else
#include "context/inc/cpu_kernel_utils.h"
#endif

namespace aicpu {
struct SparseGradient {
  float *value_;
  int *indices_;
  size_t indices_size_;
};

int Sign(float x);

struct MultiThreadComputeParams {
  float *var_;
  float *accum_;
  float *linear_;
  float *m_;
  float *m_t_;
  float *v_;
  float lr_;
  float l1_;
  float l2_;
  float lr_power_;
  float beta1_;
  float beta2_;
  float epsilon_;
  SparseGradient sparse_grad_;
  size_t var_first_dim_size_;
  size_t var_outer_dim_size_;
  bool use_nesterov_;
};

struct WorkerParamsForReduceSparseGradient {
  size_t slice_start_{0};
  size_t slice_end_{0};
  size_t max_length_{0};
  size_t outer_dim_{0};
  std::vector<std::pair<int, size_t>> *sorted_indices_{nullptr};
  std::vector<size_t> *slice_positions_{nullptr};
  float *src_value_{nullptr};
  SparseGradient *unique_grad_{nullptr};
};

void WorkerForReduceSparseGradient(CpuKernelContext &ctx, WorkerParamsForReduceSparseGradient param);

void ReduceSparseGradient(CpuKernelContext &ctx, const SparseGradient &origin_sparse_grad, SparseGradient *unique_grad,
                          size_t first_dim, size_t outer_dim);
using MultiThreadComputeFunc =
  std::function<void(CpuKernelContext &ctx, MultiThreadComputeParams *param, size_t start, size_t end)>;
uint32_t MultiThreadCompute(CpuKernelContext &ctx, const MultiThreadComputeFunc &func, MultiThreadComputeParams *params,
                            size_t total_compute_size);
void ReduceMultiSparseGradient(CpuKernelContext &ctx, std::vector<std::shared_ptr<SparseGradient>> &unique_slice_grads,
                               SparseGradient *tmp_grad, SparseGradient *unique_grad, size_t first_dim,
                               size_t outer_dim);
void TwoLevelReduceSparseGradient(CpuKernelContext &ctx, const SparseGradient &origin_sparse_grad,
                                  SparseGradient *tmp_grad, SparseGradient *unique_grad, size_t first_dim,
                                  size_t outer_dim);
}  // namespace aicpu
#endif
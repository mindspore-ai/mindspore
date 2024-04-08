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

#include "utils/fused_sparse_utils.h"
#include "inc/kernel_log.h"

namespace aicpu {
int Sign(float x) {
  if (x > 0) {
    return 1;
  }
  if (x < 0) {
    return -1;
  }
  return 0;
}

void WorkerForReduceSparseGradient(CpuKernelContext &ctx, WorkerParamsForReduceSparseGradient param) {
  auto outer_dim = param.outer_dim_;
  auto &sorted_indices = *(param.sorted_indices_);
  auto &slice_positions = *(param.slice_positions_);
  auto unique_grad = param.unique_grad_;
  for (size_t slice_id = param.slice_start_; slice_id < param.slice_end_; ++slice_id) {
    size_t cur_pos = slice_positions[slice_id];
    int index = sorted_indices[cur_pos].first;
    unique_grad->indices_[slice_id] = index;
    size_t start_index = slice_id * outer_dim;
    auto ret_code = memcpy_s(unique_grad->value_ + start_index, (param.max_length_ - start_index) * sizeof(float),
                             param.src_value_ + sorted_indices[cur_pos].second, outer_dim * sizeof(float));
    if (ret_code != EOK) {
      CUST_AICPU_LOGE(ctx, "Failed to copy data!");
    }
    cur_pos++;
    size_t end_pos;
    if (slice_id + 1 < slice_positions.size()) {
      end_pos = slice_positions[slice_id + 1];
    } else {
      end_pos = sorted_indices.size();
    }
    while (cur_pos < end_pos) {
      for (size_t i = 0; i < outer_dim; ++i) {
        unique_grad->value_[start_index + i] += param.src_value_[sorted_indices[cur_pos].second + i];
      }
      cur_pos++;
    }
  }
}

void ReduceSparseGradient(CpuKernelContext &ctx, const SparseGradient &origin_sparse_grad, SparseGradient *unique_grad,
                          size_t first_dim, size_t outer_dim) {
  std::vector<std::pair<int, size_t>> sorted_indices;
  sorted_indices.reserve(origin_sparse_grad.indices_size_);
  for (size_t i = 0; i < origin_sparse_grad.indices_size_; ++i) {
    int index = origin_sparse_grad.indices_[i];
    if (index >= 0 && static_cast<size_t>(index) < first_dim) {
      sorted_indices.emplace_back(std::pair<int, size_t>(index, i * outer_dim));
    }
  }
  std::sort(
    sorted_indices.begin(), sorted_indices.end(),
    [](const std::pair<int, size_t> &left, const std::pair<int, size_t> &right) { return left.first < right.first; });
  int last_index = 0;
  std::vector<size_t> slice_positions;
  for (size_t i = 0; i < sorted_indices.size(); ++i) {
    if (i == 0 || last_index != sorted_indices[i].first) {
      slice_positions.emplace_back(i);
    }
    last_index = sorted_indices[i].first;
  }
  size_t thread_num = 16;
  if (slice_positions.size() < thread_num) {
    thread_num = slice_positions.size();
  }
  size_t stride = (slice_positions.size() + thread_num - 1) / thread_num;
  thread_num = (slice_positions.size() + stride - 1) / stride;
  size_t max_length = sorted_indices.size() * outer_dim;
  auto shardWorkerForReduceSparseGradient = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      size_t slice_start = i * stride;
      size_t slice_end = 0;
      if (i == thread_num - 1) {
        slice_end = slice_positions.size();
      } else {
        slice_end = slice_start + stride;
      }
      WorkerParamsForReduceSparseGradient params;
      params.slice_start_ = slice_start;
      params.slice_end_ = slice_end;
      params.max_length_ = max_length;
      params.outer_dim_ = outer_dim;
      params.sorted_indices_ = &sorted_indices;
      params.slice_positions_ = &slice_positions;
      params.src_value_ = origin_sparse_grad.value_;
      params.unique_grad_ = unique_grad;

      WorkerForReduceSparseGradient(ctx, params);
    }
  };
  const int64_t per_unit_size = 1;
  CpuKernelUtils::ParallelFor(ctx, thread_num, per_unit_size, shardWorkerForReduceSparseGradient);

  unique_grad->indices_size_ = slice_positions.size();
}

using MultiThreadComputeFunc =
  std::function<void(CpuKernelContext &, MultiThreadComputeParams *param, size_t start, size_t end)>;
uint32_t MultiThreadCompute(CpuKernelContext &ctx, const MultiThreadComputeFunc &func, MultiThreadComputeParams *params,
                            size_t total_compute_size) {
  const size_t kThreadNum = 16;
  auto shardComputeFunc = [&](size_t start, size_t end) { func(ctx, params, start, end); };
  const int64_t once_compute_size = (total_compute_size + kThreadNum - 1) / kThreadNum;
  return CpuKernelUtils::ParallelFor(ctx, total_compute_size, once_compute_size, shardComputeFunc);
}

void ReduceMultiSparseGradient(CpuKernelContext &ctx, std::vector<std::shared_ptr<SparseGradient>> &unique_slice_grads,
                               SparseGradient *tmp_grad, SparseGradient *unique_grad, size_t first_dim,
                               size_t outer_dim) {
  if (unique_slice_grads.empty()) {
    return;
  }
  size_t index_data_size = outer_dim * sizeof(float);
  size_t unique_indices_size = 0;
  for (size_t i = 0; i < unique_slice_grads.size(); ++i) {
    auto &slice_grad = unique_slice_grads[i];
    auto ret_code = memcpy_s(tmp_grad->value_ + unique_indices_size * outer_dim,
                             (tmp_grad->indices_size_ - unique_indices_size) * index_data_size, slice_grad->value_,
                             slice_grad->indices_size_ * index_data_size);
    if (ret_code != EOK) {
      CUST_AICPU_LOGE(ctx, "Failed to copy data!");
    }
    ret_code =
      memcpy_s(tmp_grad->indices_ + unique_indices_size, (tmp_grad->indices_size_ - unique_indices_size) * sizeof(int),
               slice_grad->indices_, slice_grad->indices_size_ * sizeof(int));
    if (ret_code != EOK) {
      CUST_AICPU_LOGE(ctx, "Failed to copy data!");
    }
    unique_indices_size += slice_grad->indices_size_;
  }
  tmp_grad->indices_size_ = unique_indices_size;
  ReduceSparseGradient(ctx, *tmp_grad, unique_grad, first_dim, outer_dim);
}

void TwoLevelReduceSparseGradient(CpuKernelContext &ctx, const SparseGradient &origin_sparse_grad,
                                  SparseGradient *tmp_grad, SparseGradient *unique_grad, size_t first_dim,
                                  size_t outer_dim) {
  size_t thread_num = 1;
  if (origin_sparse_grad.indices_size_ < thread_num) {
    thread_num = origin_sparse_grad.indices_size_;
  }
  if (thread_num == 0) {
    return;
  }
  size_t thread_indices_size = origin_sparse_grad.indices_size_ / thread_num;
  size_t left_indices_size = origin_sparse_grad.indices_size_ % thread_num;
  std::vector<std::shared_ptr<SparseGradient>> unique_slice_grads;
  auto shardReduceSparseGradient = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {  // 0~thread_num
      size_t indices_size = thread_indices_size;
      if (i == thread_num - 1) {
        indices_size = thread_indices_size + left_indices_size;
      }
      size_t value_offset = i * thread_indices_size * outer_dim;
      size_t indices_offset = i * thread_indices_size;
      auto slice_grad = SparseGradient(
        {origin_sparse_grad.value_ + value_offset, origin_sparse_grad.indices_ + indices_offset, indices_size});
      unique_slice_grads.emplace_back(std::make_shared<SparseGradient>());
      unique_slice_grads[i]->value_ = unique_grad->value_ + value_offset;
      unique_slice_grads[i]->indices_ = unique_grad->indices_ + indices_offset;
      unique_slice_grads[i]->indices_size_ = indices_size;
      ReduceSparseGradient(ctx, slice_grad, unique_slice_grads[i].get(), first_dim, outer_dim);
    }
  };

  shardReduceSparseGradient(0, thread_num);
  ReduceMultiSparseGradient(ctx, unique_slice_grads, tmp_grad, unique_grad, first_dim, outer_dim);
}
}  // namespace aicpu
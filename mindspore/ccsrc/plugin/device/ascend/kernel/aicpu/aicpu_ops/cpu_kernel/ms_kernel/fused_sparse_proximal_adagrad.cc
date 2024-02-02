/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "fused_sparse_proximal_adagrad.h"
#include <securec.h>
#include "utils/kernel_util.h"

namespace aicpu {
namespace {
constexpr uint32_t kFusedSparseProximalAdagradInputNum = 7;
constexpr uint32_t kFusedSparseProximalAdagradOutputNum = 2;
constexpr uint32_t kVarIndex = 0;
constexpr uint32_t kAccumIndex = 1;
constexpr uint32_t kLrIndex = 2;
constexpr uint32_t kL1Index = 3;
constexpr uint32_t kL2Index = 4;
constexpr uint32_t kGradIndex = 5;
constexpr uint32_t kIndicesIndex = 6;
constexpr uint32_t kOutputVarIndex = 0;
constexpr uint32_t kOutputAccumIndex = 1;
const char *kFusedSparseProximalAdagrad = "FusedSparseProximalAdagrad";

int Sign(float x) {
  if (x > 0) {
    return 1;
  }
  if (x < 0) {
    return -1;
  }
  return 0;
}

struct SparseGradient {
  float *value_;
  int *indices_;
  size_t indices_size_;
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

void WorkerForReduceSparseGradient(WorkerParamsForReduceSparseGradient param) {
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
      AICPU_LOGE("Failed to copy data!");
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

void ComputeProximalAdagrad(MultiThreadComputeParams *input_params, size_t start, size_t end) {
  auto var = input_params->var_;
  auto accum = input_params->accum_;
  auto lr = input_params->lr_;
  auto l1 = input_params->l1_;
  auto l2 = input_params->l2_;
  auto unique_sparse_grad = input_params->sparse_grad_;
  auto var_first_dim_size = input_params->var_first_dim_size_;
  auto var_outer_dim_size = input_params->var_outer_dim_size_;
  for (size_t i = start; i < end; ++i) {
    int index = unique_sparse_grad.indices_[i];
    if (index < 0 || static_cast<size_t>(index) >= var_first_dim_size) {
      AICPU_LOGE("Index %d in indices is out of range after unique process", index);
    }
    size_t start_index = var_outer_dim_size * index;
    size_t end_index = start_index + var_outer_dim_size;
    for (size_t j = start_index, k = var_outer_dim_size * i; j < end_index; ++j, ++k) {
      auto summed_grad = unique_sparse_grad.value_[k];
      accum[j] += summed_grad * summed_grad;
      auto learning_rate = lr * (1 / std::sqrt(accum[j]));
      auto prox_v = var[j];
      prox_v -= summed_grad * learning_rate;
      if (l1 > 0) {
        var[j] = Sign(prox_v) * std::fmax(std::fabs(prox_v) - learning_rate * l1, static_cast<float>(0.0)) /
                 (1 + l2 * learning_rate);
      } else {
        var[j] = prox_v / (1 + l2 * learning_rate);
      }
    }
  }
}

void ReduceSparseGradient(const CpuKernelContext &ctx, const SparseGradient &origin_sparse_grad,
                          SparseGradient *unique_grad, size_t first_dim, size_t outer_dim) {
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

      WorkerForReduceSparseGradient(params);
    }
  };
  const int64_t per_unit_size = 1;
  CpuKernelUtils::ParallelFor(ctx, thread_num, per_unit_size, shardWorkerForReduceSparseGradient);

  unique_grad->indices_size_ = slice_positions.size();
}

using MultiThreadComputeFunc = std::function<void(MultiThreadComputeParams *param, size_t start, size_t end)>;
uint32_t MultiThreadCompute(const CpuKernelContext &ctx, const MultiThreadComputeFunc &func,
                            MultiThreadComputeParams *params, size_t total_compute_size) {
  const size_t kThreadNum = 16;
  auto shardComputeFunc = [&](size_t start, size_t end) { func(params, start, end); };
  const int64_t once_compute_size = (total_compute_size + kThreadNum - 1) / kThreadNum;
  return CpuKernelUtils::ParallelFor(ctx, total_compute_size, once_compute_size, shardComputeFunc);
}
}  // namespace

uint32_t FusedSparseProximalAdagradCpuKernel::DoCompute(CpuKernelContext &ctx) {
  // input  0~10
  auto var = reinterpret_cast<float *>(ctx.Input(kVarIndex)->GetData());
  auto accum = reinterpret_cast<float *>(ctx.Input(kAccumIndex)->GetData());
  auto lr = reinterpret_cast<float *>(ctx.Input(kLrIndex)->GetData())[0];
  auto l1 = reinterpret_cast<float *>(ctx.Input(kL1Index)->GetData())[0];
  auto l2 = reinterpret_cast<float *>(ctx.Input(kL2Index)->GetData())[0];
  auto grad = reinterpret_cast<float *>(ctx.Input(kGradIndex)->GetData());
  auto indices = reinterpret_cast<int *>(ctx.Input(kIndicesIndex)->GetData());
  // output 7~8

  float *new_grad;
  int *new_indices;
  if (indices_size_ == 0 || var_outer_dim_size_ == 0) {
    return KERNEL_STATUS_OK;
  }
  new_grad = (float *)malloc(indices_size_ * var_outer_dim_size_ * sizeof(float));
  new_indices = (int *)malloc(indices_size_ * sizeof(int));
  if (new_grad == NULL || new_indices == NULL) {
    free(new_grad);
    free(new_indices);
    new_grad = NULL;
    new_indices = NULL;
    KERNEL_LOG_ERROR("Malloc failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  SparseGradient unique_sparse_grad({new_grad, new_indices, indices_size_});
  ReduceSparseGradient(ctx, SparseGradient({grad, indices, indices_size_}), &unique_sparse_grad, var_first_dim_size_,
                       var_outer_dim_size_);

  MultiThreadComputeParams input_params;
  input_params.var_ = var;
  input_params.accum_ = accum;
  input_params.lr_ = lr;
  input_params.l1_ = l1;
  input_params.l2_ = l2;
  input_params.sparse_grad_ = unique_sparse_grad;
  input_params.var_first_dim_size_ = var_first_dim_size_;
  input_params.var_outer_dim_size_ = var_outer_dim_size_;
  KERNEL_HANDLE_ERROR(MultiThreadCompute(ctx, ComputeProximalAdagrad, &input_params, unique_sparse_grad.indices_size_),
                      "Compute worker failed.");
  free(new_grad);
  free(new_indices);
  new_grad = NULL;
  new_indices = NULL;
  return KERNEL_STATUS_OK;
}

uint32_t FusedSparseProximalAdagradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kFusedSparseProximalAdagradInputNum, kFusedSparseProximalAdagradOutputNum),
                      "Check failed.");
  auto var = ctx.Input(kVarIndex);
  auto var_shape = var->GetTensorShape()->GetDimSizes();
  auto grad = ctx.Input(kGradIndex);
  auto grad_shape = grad->GetTensorShape()->GetDimSizes();
  auto indices = ctx.Input(kIndicesIndex);
  auto indices_shape = indices->GetTensorShape()->GetDimSizes();

  for (size_t i = 0; i < kFusedSparseProximalAdagradInputNum; ++i) {
    if (i == kIndicesIndex) continue;
    auto dtype = ctx.Input(i)->GetDataType();
    if (dtype != DT_FLOAT) {
      KERNEL_LOG_ERROR("Only support data type 'float', but got [%s]", DTypeStr(dtype).c_str());
      return KERNEL_STATUS_INNER_ERROR;
    }
  }

  if (var_shape.empty()) {
    AICPU_LOGE("var must be at least 1D");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (grad_shape.empty()) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  var_first_dim_size_ = var_shape[0];
  for (size_t i = 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      AICPU_LOGE("The shape of var and grad must equal in dimension %d", i);
      return KERNEL_STATUS_INNER_ERROR;
    }
    var_outer_dim_size_ *= var_shape[i];
  }
  if (indices_shape.size() != 1) {
    AICPU_LOGE("indices must be 1D");
    return KERNEL_STATUS_INNER_ERROR;
  }
  indices_size_ = indices_shape[0];
  if (grad_shape[0] != static_cast<int64_t>(indices_size_)) {
    AICPU_LOGE("The first dimension of grad shape must be equal to indices");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return DoCompute(ctx);
}
REGISTER_MS_CPU_KERNEL(kFusedSparseProximalAdagrad, FusedSparseProximalAdagradCpuKernel);
}  // namespace aicpu

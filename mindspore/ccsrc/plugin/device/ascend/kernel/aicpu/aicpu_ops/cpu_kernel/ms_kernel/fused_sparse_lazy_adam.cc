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

#include "fused_sparse_lazy_adam.h"
#include "inc/kernel_log.h"
#include "utils/fused_sparse_utils.h"

namespace aicpu {
namespace {
const char *kFusedSparseLazyAdam = "FusedSparseLazyAdam";

void ComputeLazyAdam(CpuKernelContext &ctx, MultiThreadComputeParams *input_params, size_t start, size_t end) {
  auto var = input_params->var_;
  auto m = input_params->m_;
  auto v = input_params->v_;
  auto lr = input_params->lr_;
  auto beta1 = input_params->beta1_;
  auto beta2 = input_params->beta2_;
  auto epsilon = input_params->epsilon_;
  auto use_nesterov = input_params->use_nesterov_;
  auto unique_sparse_grad = input_params->sparse_grad_;
  auto var_first_dim_size = input_params->var_first_dim_size_;
  auto var_outer_dim_size = input_params->var_outer_dim_size_;
  for (size_t i = start; i < end; ++i) {
    int index = unique_sparse_grad.indices_[i];
    if (index < 0 || static_cast<size_t>(index) >= var_first_dim_size) {
      CUST_AICPU_LOGE(ctx, "Index %d in indices is out of range", index);
    }
    size_t start_index = var_outer_dim_size * index;
    size_t end_index = start_index + var_outer_dim_size;
    for (size_t j = start_index, k = var_outer_dim_size * i; j < end_index; ++j, ++k) {
      auto summed_grad = unique_sparse_grad.value_[k];
      m[j] = beta1 * m[j] + (1 - beta1) * summed_grad;
      v[j] = beta2 * v[j] + (1 - beta2) * summed_grad * summed_grad;
      if (use_nesterov) {
        var[j] -= lr * (m[j] * beta1 + (1 - beta1) * summed_grad) / (std::sqrt(v[j]) + epsilon);
      } else {
        var[j] -= lr * m[j] / (std::sqrt(v[j]) + epsilon);
      }
    }
  }
}
}  // namespace

uint32_t FusedSparseLazyAdamKernel::Compute(CpuKernelContext &ctx) {
  RETURN_IF_FAILURE(ParseKernelParam(ctx));
  // input  0~10
  auto var = reinterpret_cast<float *>(ctx.Input(0)->GetData());
  auto m = reinterpret_cast<float *>(ctx.Input(1)->GetData());
  auto v = reinterpret_cast<float *>(ctx.Input(2)->GetData());
  auto beta1_power = reinterpret_cast<float *>(ctx.Input(3)->GetData())[0];
  if (beta1_power == 1) {
    CUST_AICPU_LOGE(ctx, "The beta1_power should not be 1");
    return KERNEL_STATUS_INNER_ERROR;
  }
  auto beta2_power = reinterpret_cast<float *>(ctx.Input(4)->GetData())[0];
  auto lr = reinterpret_cast<float *>(ctx.Input(5)->GetData())[0];
  auto beta1 = reinterpret_cast<float *>(ctx.Input(6)->GetData())[0];
  auto beta2 = reinterpret_cast<float *>(ctx.Input(7)->GetData())[0];
  auto epsilon = reinterpret_cast<float *>(ctx.Input(8)->GetData())[0];
  auto grad = reinterpret_cast<float *>(ctx.Input(9)->GetData());
  auto indices = reinterpret_cast<int *>(ctx.Input(10)->GetData());
  // output 11~13

  float *new_grad;
  int *new_indices;
  float *tmp_grad;
  int *tmp_indices;
  if (indices_size_ == 0 || var_outer_dim_size_ == 0) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  new_grad = (float *)malloc(indices_size_ * var_outer_dim_size_ * sizeof(float));
  new_indices = (int *)malloc(indices_size_ * sizeof(int));
  tmp_grad = (float *)malloc(indices_size_ * var_outer_dim_size_ * sizeof(float));
  tmp_indices = (int *)malloc(indices_size_ * sizeof(int));
  if (new_grad == nullptr || new_indices == nullptr || tmp_grad == nullptr || tmp_indices == nullptr) {
    free(new_grad);
    free(new_indices);
    free(tmp_grad);
    free(tmp_indices);
    new_grad = nullptr;
    new_indices = nullptr;
    tmp_grad = nullptr;
    tmp_indices = nullptr;
    return KERNEL_STATUS_INNER_ERROR;
  }

  SparseGradient unique_sparse_grad({new_grad, new_indices, indices_size_});
  SparseGradient tmp_sparse_grad({tmp_grad, tmp_indices, indices_size_});
  TwoLevelReduceSparseGradient(ctx, SparseGradient({grad, indices, indices_size_}), &tmp_sparse_grad,
                               &unique_sparse_grad, var_first_dim_size_, var_outer_dim_size_);

  lr = lr * std::sqrt(1 - beta2_power) / (1 - beta1_power);
  MultiThreadComputeParams input_params;
  input_params.var_ = var;
  input_params.m_ = m;
  input_params.v_ = v;
  input_params.lr_ = lr;
  input_params.beta1_ = beta1;
  input_params.beta2_ = beta2;
  input_params.epsilon_ = epsilon;
  input_params.use_nesterov_ = use_nesterov_;
  input_params.sparse_grad_ = unique_sparse_grad;
  input_params.var_first_dim_size_ = var_first_dim_size_;
  input_params.var_outer_dim_size_ = var_outer_dim_size_;
  MultiThreadCompute(ctx, ComputeLazyAdam, &input_params, unique_sparse_grad.indices_size_);

  free(new_grad);
  free(new_indices);
  free(tmp_grad);
  free(tmp_indices);
  new_grad = nullptr;
  new_indices = nullptr;
  tmp_grad = nullptr;
  tmp_indices = nullptr;

  return KERNEL_STATUS_OK;
}

uint32_t FusedSparseLazyAdamKernel::ParseKernelParam(CpuKernelContext &ctx) {
  auto use_nesterov = ctx.GetAttr("use_nesterov");
  CUST_KERNEL_CHECK_NULLPTR(ctx, use_nesterov, KERNEL_STATUS_INNER_ERROR, "Failed to get attr 'use_nesterov'.")
  use_nesterov_ = use_nesterov->GetBool();

  auto var_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto grad_shape = ctx.Input(9)->GetTensorShape()->GetDimSizes();
  auto indices_shape = ctx.Input(10)->GetTensorShape()->GetDimSizes();

  if (var_shape.empty()) {
    CUST_AICPU_LOGE(ctx, "var must be at least 1D");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (grad_shape.empty()) {
    return KERNEL_STATUS_INNER_ERROR;
  }
  var_first_dim_size_ = var_shape[0];
  for (size_t i = 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      CUST_AICPU_LOGE(ctx, "The shape of var and grad must equal in dimension %d", i);
      return KERNEL_STATUS_INNER_ERROR;
    }
    var_outer_dim_size_ *= var_shape[i];
  }
  if (indices_shape.size() != 1) {
    CUST_AICPU_LOGE(ctx, "indices must be 1D");
    return KERNEL_STATUS_INNER_ERROR;
  }
  indices_size_ = indices_shape[0];
  if (grad_shape[0] != static_cast<int64_t>(indices_size_)) {
    CUST_AICPU_LOGE(ctx, "The first dimension of grad shape must be equal to indices");
    return KERNEL_STATUS_INNER_ERROR;
  }

  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kFusedSparseLazyAdam, FusedSparseLazyAdamKernel);
}  // namespace aicpu
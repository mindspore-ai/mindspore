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

#include "fused_sparse_ftrl.h"
#include "inc/kernel_log.h"
#include "utils/fused_sparse_utils.h"

namespace aicpu {
namespace {
const char *kFusedSparseFtrl = "FusedSparseFtrl";
const double DefaultLrPower = -0.5;

void ComputeFtrl(CpuKernelContext &ctx, MultiThreadComputeParams *input_params, size_t start, size_t end) {
  auto var = input_params->var_;
  auto accum = input_params->accum_;
  auto linear = input_params->linear_;
  auto lr = input_params->lr_;
  auto l1 = input_params->l1_;
  auto l2_plus = 2 * input_params->l2_;
  auto lr_power = input_params->lr_power_;
  auto unique_sparse_grad = input_params->sparse_grad_;
  auto var_first_dim_size = input_params->var_first_dim_size_;
  auto var_outer_dim_size = input_params->var_outer_dim_size_;
  for (size_t i = start; i < end; ++i) {
    int index = unique_sparse_grad.indices_[i];
    if (index < 0 || static_cast<size_t>(index) >= var_first_dim_size) {
      CUST_AICPU_LOGE(ctx, "Index %d in indices is out of range after unique process", index);
    }
    size_t start_index = var_outer_dim_size * index;
    size_t end_index = start_index + var_outer_dim_size;
    for (size_t j = start_index, k = var_outer_dim_size * i; j < end_index; ++j, ++k) {
      auto summed_grad = unique_sparse_grad.value_[k];
      auto accum_new = accum[j] + summed_grad * summed_grad;
      float y;
      if (lr_power == DefaultLrPower) {
        y = std::sqrt(accum_new);
        linear[j] += summed_grad - (y - std::sqrt(accum[j])) / lr * var[j];
      } else {
        y = std::pow(accum_new, -lr_power);
        linear[j] += summed_grad - (y - std::pow(accum[j], -lr_power)) / lr * var[j];
      }
      accum[j] = accum_new;
      auto x = Sign(linear[j]) * l1 - linear[j];
      y = y / lr + l2_plus;
      var[j] = std::fabs(linear[j]) > l1 ? x / y : 0;
    }
  }
}
}  // namespace

uint32_t FusedSparseFtrlKernel::Compute(CpuKernelContext &ctx) {
  RETURN_IF_FAILURE(ParseKernelParam(ctx));
  // start
  // input  0~4
  auto var = reinterpret_cast<float *>(ctx.Input(0)->GetData());
  auto accum = reinterpret_cast<float *>(ctx.Input(1)->GetData());
  auto linear = reinterpret_cast<float *>(ctx.Input(2)->GetData());
  auto grad = reinterpret_cast<float *>(ctx.Input(3)->GetData());
  auto indices = reinterpret_cast<int *>(ctx.Input(4)->GetData());
  // output 5~7

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

  SparseGradient unique_sparse_grad;
  unique_sparse_grad.value_ = new_grad;
  unique_sparse_grad.indices_ = new_indices;
  unique_sparse_grad.indices_size_ = indices_size_;

  SparseGradient tmp_sparse_grad;
  tmp_sparse_grad.value_ = tmp_grad;
  tmp_sparse_grad.indices_ = tmp_indices;
  tmp_sparse_grad.indices_size_ = indices_size_;

  SparseGradient origin_sparse_grad;
  origin_sparse_grad.value_ = grad;
  origin_sparse_grad.indices_ = indices;
  origin_sparse_grad.indices_size_ = indices_size_;

  TwoLevelReduceSparseGradient(ctx, origin_sparse_grad, &tmp_sparse_grad, &unique_sparse_grad, var_first_dim_size_,
                               var_outer_dim_size_);

  MultiThreadComputeParams input_params;
  input_params.var_ = var;
  input_params.accum_ = accum;
  input_params.linear_ = linear;
  input_params.lr_ = lr_;
  input_params.l1_ = l1_;
  input_params.l2_ = l2_;
  input_params.lr_power_ = lr_power_;
  input_params.sparse_grad_ = unique_sparse_grad;
  input_params.var_first_dim_size_ = var_first_dim_size_;
  input_params.var_outer_dim_size_ = var_outer_dim_size_;

  MultiThreadCompute(ctx, ComputeFtrl, &input_params, unique_sparse_grad.indices_size_);

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

uint32_t FusedSparseFtrlKernel::ParseKernelParam(CpuKernelContext &ctx) {
  // InitKernel
  auto lr = ctx.GetAttr("lr");
  CUST_KERNEL_CHECK_NULLPTR(ctx, lr, KERNEL_STATUS_INNER_ERROR, "Failed to get attr 'lr'.")
  auto l1 = ctx.GetAttr("l1");
  CUST_KERNEL_CHECK_NULLPTR(ctx, l1, KERNEL_STATUS_INNER_ERROR, "Failed to get attr 'l1'.")
  auto l2 = ctx.GetAttr("l2");
  CUST_KERNEL_CHECK_NULLPTR(ctx, l2, KERNEL_STATUS_INNER_ERROR, "Failed to get attr 'l2'.")
  auto lr_power = ctx.GetAttr("lr_power");
  CUST_KERNEL_CHECK_NULLPTR(ctx, lr_power, KERNEL_STATUS_INNER_ERROR, "Failed to get attr 'lr_power'.")
  lr_ = lr->GetFloat();
  l1_ = l1->GetFloat();
  l2_ = l2->GetFloat();
  lr_power_ = lr_power->GetFloat();
  if (lr_ <= 0) {
    CUST_AICPU_LOGE(ctx, "lr should be a positive scalar");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (l1_ < 0) {
    CUST_AICPU_LOGE(ctx, "l1 should be a non-negative scalar");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (l2_ < 0) {
    CUST_AICPU_LOGE(ctx, "l2 should be a non-negative scalar");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (lr_power_ > 0) {
    CUST_AICPU_LOGE(ctx, "lr_power should be a non-positive scalar");
    return KERNEL_STATUS_INNER_ERROR;
  }
  auto var_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto grad_shape = ctx.Input(3)->GetTensorShape()->GetDimSizes();
  auto indices_shape = ctx.Input(4)->GetTensorShape()->GetDimSizes();

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
REGISTER_MS_CPU_KERNEL(kFusedSparseFtrl, FusedSparseFtrlKernel);
}  // namespace aicpu
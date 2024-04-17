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
#include "context/inc/cpu_kernel_utils.h"
#include "utils/fused_sparse_utils.h"

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

void ComputeProximalAdagrad(CpuKernelContext &ctx, MultiThreadComputeParams *input_params, size_t start, size_t end) {
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
      CUST_AICPU_LOGE(ctx, "Index %d in indices is out of range after unique process", index);
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
    CUST_KERNEL_LOG_ERROR(ctx, "Malloc failed.");
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
  CUST_KERNEL_HANDLE_ERROR(
    ctx, MultiThreadCompute(ctx, ComputeProximalAdagrad, &input_params, unique_sparse_grad.indices_size_),
    "Compute worker failed.");
  free(new_grad);
  free(new_indices);
  new_grad = NULL;
  new_indices = NULL;
  return KERNEL_STATUS_OK;
}

uint32_t FusedSparseProximalAdagradCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(
    ctx, NormalCheck(ctx, kFusedSparseProximalAdagradInputNum, kFusedSparseProximalAdagradOutputNum), "Check failed.");
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
      CUST_KERNEL_LOG_ERROR(ctx, "Only support data type 'float', but got [%s]", DTypeStr(dtype).c_str());
      return KERNEL_STATUS_INNER_ERROR;
    }
  }

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
  return DoCompute(ctx);
}
REGISTER_MS_CPU_KERNEL(kFusedSparseProximalAdagrad, FusedSparseProximalAdagradCpuKernel);
}  // namespace aicpu

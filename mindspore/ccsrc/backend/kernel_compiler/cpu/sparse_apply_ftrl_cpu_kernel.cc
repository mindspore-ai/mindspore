/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/sparse_apply_ftrl_cpu_kernel.h"
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyFtrlInputsNum = 5;
constexpr size_t kSparseApplyFtrlWorkspaceSize = 4;
constexpr char kKernelName[] = "SparseApplyFtrl";

template <typename T>
void ComputeFtrl(MultiThreadComputeParams<T> *input_params, size_t start, size_t end) {
  MS_EXCEPTION_IF_NULL(input_params);
  auto var = input_params->var_;
  auto accum = input_params->accum_;
  auto linear = input_params->linear_;
  const auto lr = input_params->lr_;
  const auto l1 = input_params->l1_;
  const auto l2_plus = 2 * input_params->l2_;
  const auto lr_power = input_params->lr_power_;
  const auto unique_sparse_grad = input_params->sparse_grad_;
  const auto var_first_dim_size = input_params->var_first_dim_size_;
  const auto var_outer_dim_size = input_params->var_outer_dim_size_;
  for (size_t i = start; i < end; ++i) {
    T index = unique_sparse_grad.indices_[i];
    if (index < 0 || LongToSize(index) >= var_first_dim_size) {
      MS_LOG(EXCEPTION) << "For '" << kKernelName << "', each element in 'indices' should be in range [0, "
                        << SizeToLong(var_first_dim_size) << "), but got " << index;
    }
    size_t start_index = var_outer_dim_size * static_cast<size_t>(index);
    size_t end_index = start_index + var_outer_dim_size;
    for (size_t j = start_index, k = var_outer_dim_size * i; j < end_index; ++j, ++k) {
      auto summed_grad = unique_sparse_grad.value_[k];
      auto accum_new = accum[j] + summed_grad * summed_grad;
      float y;
      if (lr_power == -0.5) {
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

template <typename T>
void SparseApplyFtrlCpuKernelMod::InitWorkspaceSize() {
  (void)workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float));
  (void)workspace_size_list_.emplace_back(indices_size_ * sizeof(T));
  (void)workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float));
  (void)workspace_size_list_.emplace_back(indices_size_ * sizeof(T));
}

void SparseApplyFtrlCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  NativeCpuKernelMod::InitInputOutputSize(kernel_node);
  if (indices_data_type_ == kNumberTypeInt32) {
    InitWorkspaceSize<int>();
  } else if (indices_data_type_ == kNumberTypeInt64) {
    InitWorkspaceSize<int64_t>();
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'indices' should be int32 or int64, but got "
                      << TypeIdToType(indices_data_type_)->ToString();
  }
}

void SparseApplyFtrlCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> var_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  std::vector<size_t> accum_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  std::vector<size_t> linear_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
  std::vector<size_t> grad_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
  std::vector<size_t> indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);
  if (var_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'var' should be at least 1-D, but got scalar or None.";
  }
  if (!IsSameShape(var_shape, accum_shape)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'accum' should be same with the shape of 'var', "
                         "but got the shape of 'accum': "
                      << Vector2Str(accum_shape) << " and the shape of 'var': " << Vector2Str(var_shape);
  }
  if (!IsSameShape(var_shape, linear_shape)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'linear' should be same with the shape of 'var', "
                         "but got the shape of 'linear': "
                      << Vector2Str(linear_shape) << " and the shape of 'var': " << Vector2Str(var_shape);
  }
  if (var_shape.size() != grad_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'grad' should be same with the dimension of "
                         "'var', but got the dimension of 'grad': "
                      << grad_shape.size() << " and the dimension of 'var': " << var_shape.size() << ".";
  }

  var_first_dim_size_ = var_shape[0];
  for (size_t i = 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the shape of 'var' and 'grad' should equal in dimension i=" << i
                        << ", but got 'var_shape[i]': " << var_shape[i] << " and 'grad_shape[i]': " << grad_shape[i];
    }
    var_outer_dim_size_ *= var_shape[i];
  }
  if (indices_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'indices' should be a 1-D vector, but got "
                      << indices_shape.size() << "-D.";
  }
  indices_size_ = indices_shape[0];
  if (grad_shape[0] != indices_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the first dimension value of 'grad' should be equal to "
                         "the first dimension value of 'indices', but got the first dimension value of 'grad': "
                      << grad_shape[0] << ", and the first dimension value of 'indices': " << indices_size_;
  }
  lr_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "lr");
  if (lr_ <= 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'lr' should be a positive scalar, but got " << lr_;
  }
  l1_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "l1");
  if (l1_ < 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'l1' should be a non-negative scalar, but got " << l1_;
  }
  l2_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "l2");
  if (l2_ < 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'l2' should be a non-negative scalar, but got " << l2_;
  }
  lr_power_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "lr_power");
  if (lr_power_ > 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'lr_power' should be a non-negative scalar, but got "
                      << lr_power_;
  }
  indices_data_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 4);
}

template <typename T>
void SparseApplyFtrlCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &workspace) const {
  auto *var = reinterpret_cast<float *>(inputs[0]->addr);
  auto *accum = reinterpret_cast<float *>(inputs[1]->addr);
  auto *linear = reinterpret_cast<float *>(inputs[2]->addr);
  auto *grad = reinterpret_cast<float *>(inputs[3]->addr);
  auto *indices = reinterpret_cast<T *>(inputs[4]->addr);
  auto *new_grad = reinterpret_cast<float *>(workspace[0]->addr);
  auto *new_indices = reinterpret_cast<T *>(workspace[1]->addr);
  auto *workspace_grad = reinterpret_cast<float *>(workspace[2]->addr);
  auto *workspace_indices = reinterpret_cast<T *>(workspace[3]->addr);

  SparseGradient<T> unique_sparse_grad({new_grad, new_indices, indices_size_});
  SparseGradient<T> workspace_sparse_grad({workspace_grad, workspace_indices, indices_size_});
  SparseGradient<T> input_sparse_grad({grad, indices, indices_size_});
  ReduceSparseGradientParam<T> param;
  param.input_grad_ = &input_sparse_grad;
  param.workspace_grad_ = &workspace_sparse_grad;
  param.output_grad_ = &unique_sparse_grad;
  param.max_index_ = var_first_dim_size_;
  param.value_stride_ = var_outer_dim_size_;
  BucketReduceSparseGradient(param);

  MultiThreadComputeParams<T> input_params;
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
  MultiThreadCompute<T>(ComputeFtrl<T>, &input_params, unique_sparse_grad.indices_size_);
}

bool SparseApplyFtrlCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &workspace,
                                         const std::vector<kernel::AddressPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseApplyFtrlInputsNum, kernel_name_);
  CHECK_KERNEL_WORKSPACE_SIZE(workspace.size(), kSparseApplyFtrlWorkspaceSize, kernel_name_);
  if (indices_data_type_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, workspace);
  } else if (indices_data_type_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, workspace);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'indices' should be int32 or int64, but got "
                      << TypeIdToType(indices_data_type_)->ToString();
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore

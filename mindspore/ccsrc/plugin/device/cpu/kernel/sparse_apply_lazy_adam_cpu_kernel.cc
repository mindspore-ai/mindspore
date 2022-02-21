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

#include "plugin/device/cpu/kernel/sparse_apply_lazy_adam_cpu_kernel.h"
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyLazyAdamInputsNum = 11;
constexpr size_t kSparseApplyLazyAdamWorkspaceSize = 4;
constexpr char kKernelName[] = "SparseApplyLazyAdam";

template <typename T>
void ComputeLazyAdam(MultiThreadComputeParams<T> *input_params, size_t start, size_t end) {
  MS_EXCEPTION_IF_NULL(input_params);
  auto var = input_params->var_;
  auto m = input_params->m_;
  auto v = input_params->v_;
  const auto lr = input_params->lr_;
  const auto beta1 = input_params->beta1_;
  const auto beta2 = input_params->beta2_;
  const auto epsilon = input_params->epsilon_;
  const auto use_nesterov = input_params->use_nesterov_;
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

template <typename T>
void SparseApplyLazyAdamCpuKernelMod::InitWorkspaceSize() {
  (void)workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float));
  (void)workspace_size_list_.emplace_back(indices_size_ * sizeof(T));
  (void)workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float));
  (void)workspace_size_list_.emplace_back(indices_size_ * sizeof(T));
}

void SparseApplyLazyAdamCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  NativeCpuKernelMod::InitInputOutputSize(kernel_node);
  if (indices_data_type_ == kNumberTypeInt32) {
    InitWorkspaceSize<int>();
  } else if (indices_data_type_ == kNumberTypeInt64) {
    InitWorkspaceSize<int64_t>();
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'indices' should be int32 or int64,  but got "
                      << TypeIdToType(indices_data_type_)->ToString();
  }
}

void SparseApplyLazyAdamCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> var_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  std::vector<size_t> m_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  std::vector<size_t> v_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
  std::vector<size_t> grad_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 9);
  std::vector<size_t> indices_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 10);
  if (var_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'var' should be at least 1-D, but got scalar or None.";
  }
  if (!IsSameShape(var_shape, m_shape)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'm' should be same with the shape of 'var', but got the shape of 'm': "
                      << Vector2Str(m_shape) << " and the shape of 'var': " << Vector2Str(var_shape);
  }
  if (!IsSameShape(var_shape, v_shape)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'v' should be same with the shape of 'var', but got the shape of 'v': "
                      << Vector2Str(v_shape) << " and the shape of 'var': " << Vector2Str(var_shape);
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
  if (common::AnfAlgo::HasNodeAttr(USE_NESTEROV, kernel_node)) {
    use_nesterov_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, USE_NESTEROV);
  }
  indices_data_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 10);
}

template <typename T>
void SparseApplyLazyAdamCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                   const std::vector<kernel::AddressPtr> &workspace) const {
  auto *var = reinterpret_cast<float *>(inputs[0]->addr);
  auto *m = reinterpret_cast<float *>(inputs[1]->addr);
  auto *v = reinterpret_cast<float *>(inputs[2]->addr);
  auto beta1_power = reinterpret_cast<float *>(inputs[3]->addr)[0];
  if (beta1_power == 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'beta1_power' should not be 1.";
  }
  auto beta2_power = reinterpret_cast<float *>(inputs[4]->addr)[0];
  auto lr = reinterpret_cast<float *>(inputs[5]->addr)[0];
  auto beta1 = reinterpret_cast<float *>(inputs[6]->addr)[0];
  auto beta2 = reinterpret_cast<float *>(inputs[7]->addr)[0];
  auto epsilon = reinterpret_cast<float *>(inputs[8]->addr)[0];
  auto *grad = reinterpret_cast<float *>(inputs[9]->addr);
  auto *indices = reinterpret_cast<T *>(inputs[10]->addr);
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

  lr = lr * std::sqrt(1 - beta2_power) / (1 - beta1_power);
  MultiThreadComputeParams<T> input_params;
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
  MultiThreadCompute<T>(ComputeLazyAdam<T>, &input_params, unique_sparse_grad.indices_size_);
}

bool SparseApplyLazyAdamCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &workspace,
                                             const std::vector<kernel::AddressPtr> &) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseApplyLazyAdamInputsNum, kernel_name_);
  CHECK_KERNEL_WORKSPACE_SIZE(workspace.size(), kSparseApplyLazyAdamWorkspaceSize, kernel_name_);
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

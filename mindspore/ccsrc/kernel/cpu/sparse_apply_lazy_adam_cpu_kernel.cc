/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "kernel/cpu/sparse_apply_lazy_adam_cpu_kernel.h"
#include "kernel/common_utils.h"
#include "device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyLazyAdamInputSize = 11;
}  // namespace

void SparseApplyLazyAdamCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_node);
  workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float));
  workspace_size_list_.emplace_back(indices_size_ * sizeof(int));
}

void SparseApplyLazyAdamCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> var_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  std::vector<size_t> m_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  std::vector<size_t> v_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
  std::vector<size_t> grad_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 9);
  std::vector<size_t> indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 10);
  if (!IsSameShape(var_shape, m_shape)) {
    MS_LOG(EXCEPTION) << "var and m should have the same shape";
  }
  if (!IsSameShape(var_shape, v_shape)) {
    MS_LOG(EXCEPTION) << "var and v should have the same shape";
  }
  if (var_shape.empty()) {
    MS_LOG(EXCEPTION) << "var must be at least 1D";
  }
  var_first_dim_size_ = var_shape[0];
  for (size_t i = 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      MS_LOG(EXCEPTION) << "The shape of var and grad must equal in dimension " << i;
    }
    var_outer_dim_size_ *= var_shape[i];
  }
  if (indices_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "indices must be 1D";
  }
  indices_size_ = indices_shape[0];
  if (grad_shape[0] != indices_size_) {
    MS_LOG(EXCEPTION) << "The first dimension of grad shape must be equal to indices";
  }
  if (AnfAlgo::HasNodeAttr(USE_NESTEROV, kernel_node)) {
    use_nesterov_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, "use_nesterov");
  }
}

bool SparseApplyLazyAdamCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &workspace,
                                          const std::vector<kernel::AddressPtr> & /*outputs*/) {
  if (inputs.size() < kSparseApplyLazyAdamInputSize) {
    MS_LOG(EXCEPTION) << "Error input size!";
  }

  auto var = reinterpret_cast<float *>(inputs[0]->addr);
  auto m = reinterpret_cast<float *>(inputs[1]->addr);
  auto v = reinterpret_cast<float *>(inputs[2]->addr);
  auto beta1_power = reinterpret_cast<float *>(inputs[3]->addr)[0];
  if (beta1_power == 1) {
    MS_LOG(EXCEPTION) << "The beta1_power should not be 1";
  }
  auto beta2_power = reinterpret_cast<float *>(inputs[4]->addr)[0];
  auto lr = reinterpret_cast<float *>(inputs[5]->addr)[0];
  auto beta1 = reinterpret_cast<float *>(inputs[6]->addr)[0];
  auto beta2 = reinterpret_cast<float *>(inputs[7]->addr)[0];
  auto epsilon = reinterpret_cast<float *>(inputs[8]->addr)[0];
  auto grad = reinterpret_cast<float *>(inputs[9]->addr);
  auto indices = reinterpret_cast<int *>(inputs[10]->addr);
  auto new_grad = reinterpret_cast<float *>(workspace[0]->addr);
  auto new_indices = reinterpret_cast<int *>(workspace[1]->addr);

  SparseGradient unique_sparse_grad({new_grad, new_indices, indices_size_});
  ReduceSparseGradient(SparseGradient({grad, indices, indices_size_}), &unique_sparse_grad, var_first_dim_size_,
                       var_outer_dim_size_);

  lr = lr * std::sqrt(1 - beta2_power) / (1 - beta1_power);
  for (size_t i = 0; i < unique_sparse_grad.indices_size_; ++i) {
    int index = unique_sparse_grad.indices_[i];
    if (index < 0 || IntToSize(index) >= var_first_dim_size_) {
      MS_LOG(EXCEPTION) << "Index " << index << " in indices is out of range";
    }
    size_t start_index = var_outer_dim_size_ * index;
    size_t end_index = start_index + var_outer_dim_size_;
    for (size_t j = start_index, k = var_outer_dim_size_ * i; j < end_index; ++j, ++k) {
      auto summed_grad = unique_sparse_grad.value_[k];
      m[j] = beta1 * m[j] + (1 - beta1) * summed_grad;
      v[j] = beta2 * v[j] + (1 - beta2) * summed_grad * summed_grad;
      if (use_nesterov_) {
        var[j] -= lr * (m[j] * beta1 + (1 - beta1) * summed_grad) / (std::sqrt(v[j]) + epsilon);
      } else {
        var[j] -= lr * m[j] / (std::sqrt(v[j]) + epsilon);
      }
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore

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
#include "kernel/cpu/sparse_apply_proximal_adagrad_cpu_kernel.h"
#include "kernel/common_utils.h"
#include "device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyProximalAdagradInputSize = 7;
}  // namespace

void SparseApplyProximalAdagradCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_node);
  workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float));
  workspace_size_list_.emplace_back(indices_size_ * sizeof(int));
}

void SparseApplyProximalAdagradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> var_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  std::vector<size_t> accum_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  std::vector<size_t> lr_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
  std::vector<size_t> l1_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
  std::vector<size_t> l2_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);
  std::vector<size_t> grad_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 5);
  std::vector<size_t> indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 6);
  if (!IsSameShape(var_shape, accum_shape)) {
    MS_LOG(EXCEPTION) << "var and accum should have the same shape";
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
    MS_LOG(EXCEPTION) << "indices must be a 1D vector";
  }
  indices_size_ = indices_shape[0];
  if (grad_shape[0] != indices_size_) {
    MS_LOG(EXCEPTION) << "The first dimension of grad shape must be equal to indices";
  }
  if (!lr_shape.empty()) {
    MS_LOG(EXCEPTION) << "lr is not a scalar";
  }
  if (!l1_shape.empty()) {
    MS_LOG(EXCEPTION) << "l1 is not a scalar";
  }
  if (!l2_shape.empty()) {
    MS_LOG(EXCEPTION) << "l2 is not a scalar";
  }
}

bool SparseApplyProximalAdagradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &workspace,
                                                 const std::vector<kernel::AddressPtr> & /*outputs*/) {
  if (inputs.size() < kSparseApplyProximalAdagradInputSize) {
    MS_LOG(EXCEPTION) << "Wrong input size!";
  }

  auto var = reinterpret_cast<float *>(inputs[0]->addr);
  auto accum = reinterpret_cast<float *>(inputs[1]->addr);
  auto lr = reinterpret_cast<float *>(inputs[2]->addr)[0];
  auto l1 = reinterpret_cast<float *>(inputs[3]->addr)[0];
  auto l2 = reinterpret_cast<float *>(inputs[4]->addr)[0];
  auto grad = reinterpret_cast<float *>(inputs[5]->addr);
  auto indices = reinterpret_cast<int *>(inputs[6]->addr);
  auto new_grad = reinterpret_cast<float *>(workspace[0]->addr);
  auto new_indices = reinterpret_cast<int *>(workspace[1]->addr);
  SparseGradient unique_sparse_grad({new_grad, new_indices, indices_size_});
  ReduceSparseGradient(SparseGradient({grad, indices, indices_size_}), &unique_sparse_grad, var_first_dim_size_,
                       var_outer_dim_size_);

  for (size_t i = 0; i < unique_sparse_grad.indices_size_; ++i) {
    int index = unique_sparse_grad.indices_[i];
    if (index < 0 || IntToSize(index) >= var_first_dim_size_) {
      MS_LOG(EXCEPTION) << "Index " << index << " in indices is out of range after unique process";
    }
    size_t start_index = var_outer_dim_size_ * index;
    size_t end_index = start_index + var_outer_dim_size_;
    for (size_t j = start_index, k = var_outer_dim_size_ * i; j < end_index; ++j, ++k) {
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
  return true;
}
}  // namespace kernel
}  // namespace mindspore

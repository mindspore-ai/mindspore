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
#include "kernel/cpu/sparse_apply_adam_cpu_kernel.h"
#include "kernel/common_utils.h"
#include "device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseApplyAdamInputSize = 11;

void ComputeAdam(MultiThreadComputeParams *input_params, size_t start, size_t end) {
  MS_EXCEPTION_IF_NULL(input_params);
  auto m = input_params->m_;
  auto m_t = input_params->m_t_;
  auto v = input_params->v_;
  auto beta1 = input_params->beta1_;
  auto beta2 = input_params->beta2_;
  auto use_nesterov = input_params->use_nesterov_;
  auto unique_sparse_grad = input_params->sparse_grad_;
  auto var_first_dim_size = input_params->var_first_dim_size_;
  auto var_outer_dim_size = input_params->var_outer_dim_size_;
  for (size_t i = start; i < end; ++i) {
    int index = unique_sparse_grad.indices_[i];
    if (index < 0 || IntToSize(index) >= var_first_dim_size) {
      MS_LOG(EXCEPTION) << "Index " << index << " in indices is out of range after unique process";
    }
    size_t start_index = var_outer_dim_size * index;
    size_t end_index = start_index + var_outer_dim_size;
    for (size_t j = start_index, k = var_outer_dim_size * i; j < end_index; ++j, ++k) {
      auto summed_grad = unique_sparse_grad.value_[k];
      m[j] += (1 - beta1) * summed_grad;
      v[j] += (1 - beta2) * summed_grad * summed_grad;
      if (use_nesterov) {
        m_t[j] = m[j] * beta1 + (1 - beta1) * summed_grad;
      }
    }
  }
}

void ComputeMomentum(MultiThreadComputeParams *input_params, size_t start, size_t end) {
  MS_EXCEPTION_IF_NULL(input_params);
  auto m = input_params->m_;
  auto v = input_params->v_;
  auto beta1 = input_params->beta1_;
  auto beta2 = input_params->beta2_;
  for (size_t i = start; i < end; ++i) {
    m[i] *= beta1;
    v[i] *= beta2;
  }
}

void ComputeWeight(MultiThreadComputeParams *input_params, size_t start, size_t end) {
  MS_EXCEPTION_IF_NULL(input_params);
  auto var = input_params->var_;
  auto m = input_params->m_;
  auto v = input_params->v_;
  auto lr = input_params->lr_;
  auto epsilon = input_params->epsilon_;
  for (size_t i = start; i < end; ++i) {
    var[i] -= lr * m[i] / (std::sqrt(v[i]) + epsilon);
  }
}
}  // namespace

void SparseApplyAdamCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_node);
  workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float));
  workspace_size_list_.emplace_back(indices_size_ * sizeof(int));
}

void SparseApplyAdamCPUKernel::InitKernel(const CNodePtr &kernel_node) {
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

bool SparseApplyAdamCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &workspace,
                                      const std::vector<kernel::AddressPtr> & /*outputs*/) {
  if (inputs.size() < kSparseApplyAdamInputSize) {
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
  size_t total_dim_size = var_first_dim_size_ * var_outer_dim_size_;
  lr = lr * std::sqrt(1 - beta2_power) / (1 - beta1_power);

  MultiThreadComputeParams input_params;
  input_params.m_ = m;
  input_params.v_ = v;
  input_params.beta1_ = beta1;
  input_params.beta2_ = beta2;
  const size_t kThreadNum = 16;
  MultiThreadCompute(ComputeMomentum, &input_params, kThreadNum, total_dim_size);

  std::vector<float> m_t(m, m + total_dim_size);
  input_params.m_t_ = m_t.data();
  input_params.use_nesterov_ = use_nesterov_;
  input_params.sparse_grad_ = unique_sparse_grad;
  input_params.var_first_dim_size_ = var_first_dim_size_;
  input_params.var_outer_dim_size_ = var_outer_dim_size_;
  MultiThreadCompute(ComputeAdam, &input_params, kThreadNum, unique_sparse_grad.indices_size_);

  if (use_nesterov_) {
    input_params.m_ = input_params.m_t_;
  }
  input_params.var_ = var;
  input_params.lr_ = lr;
  input_params.epsilon_ = epsilon;
  MultiThreadCompute(ComputeWeight, &input_params, kThreadNum, total_dim_size);
  return true;
}
}  // namespace kernel
}  // namespace mindspore

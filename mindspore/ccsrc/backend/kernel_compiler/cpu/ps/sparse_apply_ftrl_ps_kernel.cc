/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/ps/sparse_apply_ftrl_ps_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace ps {
constexpr size_t kSparseApplyFtrlPSInputSize = 5;

void SparseApplyFtrlPSKernel::InitKernel(
  const CNodePtr &cnode, const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &shapes) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(shapes);
  const std::vector<std::shared_ptr<std::vector<size_t>>> &shape_vec = *shapes;
  if (shape_vec.size() < kSparseApplyFtrlPSInputSize) {
    MS_LOG(EXCEPTION) << "SparseApplyAdamPSKernel needs " << kSparseApplyFtrlPSInputSize << " input shapes, but got "
                      << shape_vec.size();
  }
  std::vector<size_t> var_shape = *(shape_vec[0]);
  std::vector<size_t> accum_shape = *(shape_vec[1]);
  std::vector<size_t> linear_shape = *(shape_vec[2]);
  std::vector<size_t> grad_shape = *(shape_vec[3]);
  std::vector<size_t> indices_shape = *(shape_vec[4]);

  Shard(&var_shape, 0);
  Shard(&accum_shape, 0);
  Shard(&linear_shape, 0);

  if (var_shape.size() != grad_shape.size()) {
    MS_LOG(EXCEPTION) << "var and grad should have the same shape size";
  }
  if (var_shape.empty()) {
    MS_LOG(EXCEPTION) << "var must be at least 1D";
  } else {
    var_first_dim_size_ = var_shape[0];
  }

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
  init_accum_ = AnfAlgo::GetNodeAttr<float>(cnode, "init_accum");
  if (init_accum_ < 0) {
    MS_LOG(EXCEPTION) << "init_accum should be a non-negative scalar";
  }
  lr_ = AnfAlgo::GetNodeAttr<float>(cnode, "lr");
  if (lr_ <= 0) {
    MS_LOG(EXCEPTION) << "lr should be a positive scalar";
  }
  l1_ = AnfAlgo::GetNodeAttr<float>(cnode, "l1");
  if (l1_ < 0) {
    MS_LOG(EXCEPTION) << "l1 should be a non-negative scalar";
  }
  l2_ = AnfAlgo::GetNodeAttr<float>(cnode, "l2");
  if (l2_ < 0) {
    MS_LOG(EXCEPTION) << "l2 should be a non-negative scalar";
  }
  lr_power_ = AnfAlgo::GetNodeAttr<float>(cnode, "lr_power");
  if (lr_power_ > 0) {
    MS_LOG(EXCEPTION) << "lr_power should be a non-positive scalar";
  }
  (void)workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float) * worker_num_);
  (void)workspace_size_list_.emplace_back(indices_size_ * sizeof(int) * worker_num_);
  (void)workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float) * worker_num_);
  (void)workspace_size_list_.emplace_back(indices_size_ * sizeof(int) * worker_num_);
}

void SparseApplyFtrlPSKernel::ReInit(const std::vector<std::vector<size_t>> &shapes) {
  if (shapes.empty() || shapes[0].empty()) {
    MS_LOG(EXCEPTION) << "Shape should not empty";
  }
  const std::vector<size_t> &indices_shape = shapes[0];
  indices_size_ = indices_shape[0];
  workspace_size_list_[0] = indices_size_ * var_outer_dim_size_ * sizeof(float) * worker_num_;
  workspace_size_list_[1] = indices_size_ * sizeof(int) * worker_num_;
}

void SparseApplyFtrlPSKernel::ReInit(const std::vector<AddressPtr> &inputs) {
  if (inputs.size() < kSparseApplyFtrlPSInputSize) {
    MS_LOG(EXCEPTION) << "Input numbers should not less than " << kSparseApplyFtrlPSInputSize << ", but got "
                      << inputs.size();
  }
  const auto &indices_addr = inputs[4];
  indices_size_ = indices_addr->size / sizeof(int);
  workspace_size_list_[0] = indices_size_ * var_outer_dim_size_ * sizeof(float) * worker_num_;
  workspace_size_list_[1] = indices_size_ * sizeof(int) * worker_num_;
}

bool SparseApplyFtrlPSKernel::Execute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  ReInit(inputs);
  if (indices_size_ == 0) {
    return true;
  }
  return Launch(inputs, workspace, outputs);
}

const std::vector<size_t> &SparseApplyFtrlPSKernel::input_sizes() const { return GetInputSizeList(); }

const std::vector<size_t> &SparseApplyFtrlPSKernel::output_sizes() const { return GetOutputSizeList(); }

const std::vector<size_t> &SparseApplyFtrlPSKernel::workspace_sizes() const { return GetWorkspaceSizeList(); }
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore

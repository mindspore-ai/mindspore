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
#include "backend/kernel_compiler/cpu/ps/sparse_apply_lazy_adam_ps_kernel.h"
#include <memory>
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "ps/util.h"

namespace mindspore {
namespace kernel {
namespace ps {
void SparseApplyLazyAdamPSKernel::InitKernel(
  const CNodePtr &cnode, const std::shared_ptr<std::vector<std::shared_ptr<std::vector<size_t>>>> &shapes) {
  const std::vector<std::shared_ptr<std::vector<size_t>>> &shape_vec = *shapes;
  std::vector<size_t> &var_shape = *(shape_vec[0]);
  std::vector<size_t> &m_shape = *(shape_vec[1]);
  std::vector<size_t> &v_shape = *(shape_vec[2]);
  const std::vector<size_t> &grad_shape = *(shape_vec[9]);
  const std::vector<size_t> &indices_shape = *(shape_vec[10]);

  Shard(&var_shape, 0);
  Shard(&m_shape, 0);
  Shard(&v_shape, 0);

  if (!IsSameShape(var_shape, m_shape)) {
    MS_LOG(EXCEPTION) << "var and m should have the same shape";
  }
  if (!IsSameShape(var_shape, v_shape)) {
    MS_LOG(EXCEPTION) << "var and v should have the same shape";
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
    MS_LOG(ERROR) << "The first dimension of grad shape must be equal to indices";
  }
  if (AnfAlgo::HasNodeAttr(USE_NESTEROV, cnode)) {
    use_nesterov_ = AnfAlgo::GetNodeAttr<bool>(cnode, "use_nesterov");
  }
  workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float) * worker_num_);
  workspace_size_list_.emplace_back(indices_size_ * sizeof(int) * worker_num_);
  workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float) * worker_num_);
  workspace_size_list_.emplace_back(indices_size_ * sizeof(int) * worker_num_);
}

void SparseApplyLazyAdamPSKernel::ReInit(const std::vector<std::vector<size_t>> &shapes) {
  const std::vector<size_t> &indices_shape = shapes[0];
  indices_size_ = indices_shape[0];
  workspace_size_list_[0] = indices_size_ * var_outer_dim_size_ * sizeof(float) * worker_num_;
  workspace_size_list_[1] = indices_size_ * sizeof(int) * worker_num_;
}

void SparseApplyLazyAdamPSKernel::ReInit(const std::vector<AddressPtr> &inputs) {
  const auto &indices_addr = inputs[10];
  indices_size_ = indices_addr->size / sizeof(int);
  workspace_size_list_[0] = indices_size_ * var_outer_dim_size_ * sizeof(float) * worker_num_;
  workspace_size_list_[1] = indices_size_ * sizeof(int) * worker_num_;
}

bool SparseApplyLazyAdamPSKernel::Execute(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs) {
  ReInit(inputs);
  if (indices_size_ == 0) {
    return true;
  }
  return Launch(inputs, workspace, outputs);
}

const std::vector<size_t> &SparseApplyLazyAdamPSKernel::input_sizes() const { return GetInputSizeList(); }

const std::vector<size_t> &SparseApplyLazyAdamPSKernel::output_sizes() const { return GetOutputSizeList(); }

const std::vector<size_t> &SparseApplyLazyAdamPSKernel::workspace_sizes() const { return GetWorkspaceSizeList(); }
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore

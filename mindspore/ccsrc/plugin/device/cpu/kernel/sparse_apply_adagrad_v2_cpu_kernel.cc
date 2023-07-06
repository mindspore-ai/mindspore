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

#include "plugin/device/cpu/kernel/sparse_apply_adagrad_v2_cpu_kernel.h"
#include <functional>
#include <memory>
#include <map>
#include <utility>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "ops/sparse_apply_adagrad_v2.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kVarIndex = 0;
constexpr size_t kAccumIndex = 1;
constexpr size_t kGradIndex = 2;
constexpr size_t kIndicesIndex = 3;
constexpr size_t kSparseApplyAdagradV2InputsNum = 4;
constexpr size_t kSparseApplyAdagradV2WorkspaceSize = 4;
constexpr char kKernelName[] = "SparseApplyAdagradV2";

using KernelRunFunc = SparseApplyAdagradV2CpuKernelMod::KernelRunFunc;
}  // namespace

template <typename T>
void SparseApplyAdagradV2CpuKernelMod::InitWorkspaceSize() {
  (void)workspace_size_list_.emplace_back(batch_size_ * indices_size_ * var_outer_dim_size_ * sizeof(float));
  (void)workspace_size_list_.emplace_back(batch_size_ * indices_size_ * sizeof(T));
  (void)workspace_size_list_.emplace_back(batch_size_ * indices_size_ * var_outer_dim_size_ * sizeof(float));
  (void)workspace_size_list_.emplace_back(batch_size_ * indices_size_ * sizeof(T));
}

bool SparseApplyAdagradV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != kSparseApplyAdagradV2InputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input size must be " << kSparseApplyAdagradV2InputsNum
                  << ", but got " << inputs.size();
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::SparseApplyAdagradV2>(base_operator->GetPrim());
  lr_ = kernel_ptr->get_lr();
  epsilon_ = kernel_ptr->get_epsilon();
  update_slots_ = kernel_ptr->get_update_slots();
  batch_rank_ = base_operator->get_batch_rank();
  if (lr_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', 'lr' must be a positive scalar, but got " << lr_;
    return false;
  }
  if (epsilon_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', 'epsilon' must be a positive scalar, but got " << epsilon_;
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

void SparseApplyAdagradV2CpuKernelMod::ResetResource() noexcept {
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  indices_data_type_ = kNumberTypeInt32;
  indices_size_ = 0;
  var_first_dim_size_ = 0;
  var_outer_dim_size_ = 1;
}

int SparseApplyAdagradV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  ShapeVector var_shape = inputs[kVarIndex]->GetShapeVector();
  ShapeVector accum_shape = inputs[kAccumIndex]->GetShapeVector();
  ShapeVector grad_shape = inputs[kGradIndex]->GetShapeVector();
  ShapeVector indices_shape = inputs[kIndicesIndex]->GetShapeVector();
  if (batch_rank_ > 0) {
    batch_size_ = std::accumulate(var_shape.begin(), var_shape.begin() + batch_rank_, 1, std::multiplies<int64_t>());
    if (batch_size_ == 0) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
      return KRET_RESIZE_FAILED;
    }
    var_inner_size_ =
      std::accumulate(var_shape.begin() + batch_rank_, var_shape.end(), size_t(1), std::multiplies<size_t>());
    indices_inner_size_ =
      std::accumulate(indices_shape.begin() + batch_rank_, indices_shape.end(), size_t(1), std::multiplies<size_t>());
    grad_inner_size_ =
      std::accumulate(grad_shape.begin() + batch_rank_, grad_shape.end(), size_t(1), std::multiplies<size_t>());
  }
  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(var_shape, accum_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'accum' must be the same as the shape of 'var', "
                     "but got the shape of 'accum': "
                  << accum_shape << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }
  if (var_shape.size() != grad_shape.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'grad' must be the same as the dimension of "
                     "'var', but got the dimension of 'grad': "
                  << grad_shape.size() << " and the dimension of 'var': " << var_shape.size() << ".";
    return KRET_RESIZE_FAILED;
  }
  var_first_dim_size_ = var_shape[batch_rank_];
  for (size_t i = batch_rank_ + 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of 'var' and 'grad' must be equal in dimension i=" << i
                    << ", but got 'var_shape[i]': " << var_shape[i] << " and 'grad_shape[i]': " << grad_shape[i];
      return KRET_RESIZE_FAILED;
    }
    var_outer_dim_size_ *= var_shape[i];
  }
  if (indices_shape.size() != LongToSize(batch_rank_ + 1)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'indices' must be a " << (batch_rank_ + 1)
                  << "-D vector, but got " << indices_shape.size() << "-D.";
    return KRET_RESIZE_FAILED;
  }
  indices_size_ = indices_shape[batch_rank_];
  if (grad_shape[batch_rank_] != SizeToLong(indices_size_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the first dimension value of 'grad' must be equal to "
                     "the first dimension value of 'indices', but got the first dimension value of 'grad': "
                  << grad_shape[batch_rank_] << ", and the first dimension value of 'indices': " << indices_size_;
    return KRET_RESIZE_FAILED;
  }
  indices_data_type_ = inputs[kIndicesIndex]->GetDtype();
  if (indices_data_type_ == kNumberTypeInt32) {
    InitWorkspaceSize<int>();
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dtype of 'indices' must be int32 but got "
                  << TypeIdToType(indices_data_type_)->ToString();

    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SparseApplyAdagradV2CpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutInRef(0, 0)
       .AddOutInRef(1, 1),
     &SparseApplyAdagradV2CpuKernelMod::LaunchKernel<int>}};
  return func_list;
}

template <typename T>
bool SparseApplyAdagradV2CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                    const std::vector<kernel::AddressPtr> &workspace,
                                                    const std::vector<kernel::AddressPtr> &) const {
  auto *var = reinterpret_cast<float *>(inputs[0]->addr);
  auto *accum = reinterpret_cast<float *>(inputs[1]->addr);
  auto *grad = reinterpret_cast<float *>(inputs[2]->addr);
  auto *indices = reinterpret_cast<T *>(inputs[3]->addr);

  for (int64_t index = 0; index < batch_size_; index++) {
    SparseGradient<T> input_sparse_grad({grad, indices, indices_size_});
    const auto lr = lr_;
    const auto epsilon = lr_;
    const auto update_slots = update_slots_;
    const auto unique_sparse_grad = input_sparse_grad;
    const auto var_first_dim_size = var_first_dim_size_;
    const auto var_outer_dim_size = var_outer_dim_size_;
    auto task = [this, &var, &accum, lr, epsilon, update_slots, &unique_sparse_grad, var_first_dim_size,
                 var_outer_dim_size](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        T index = unique_sparse_grad.indices_[i];
        if (index < 0 || LongToSize(index) >= var_first_dim_size) {
          MS_LOG(EXCEPTION) << "For '" << kKernelName << "', each element in 'indices' must be in range [0, "
                            << SizeToLong(var_first_dim_size) << "), but got " << index;
        }
        size_t start_index = var_outer_dim_size * static_cast<size_t>(index);
        size_t end_index = start_index + var_outer_dim_size;
        for (size_t j = start_index, k = var_outer_dim_size * i; j < end_index; ++j, ++k) {
          auto summed_grad = unique_sparse_grad.value_[k];
          if (update_slots) {
            accum[j] += summed_grad * summed_grad;
          }
          auto learning_rate = lr * (1 / std::sqrt(accum[j] + epsilon));
          var[j] -= summed_grad * learning_rate;
        }
      }
    };
    ParallelLaunch(task, indices_size_, 0);
    // apply offset to all address pointers.
    var += var_inner_size_;
    accum += var_inner_size_;
    grad += grad_inner_size_;
    indices += indices_inner_size_;
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseApplyAdagradV2, SparseApplyAdagradV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/nn/ctcloss_v2_grad_gpu_kernel.h"
#include <memory>
#include "mindspore/ops/infer/ctc_loss_v2_grad.h"
namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = CTCLossV2GradGpuKernelMod::KernelRunFunc;
constexpr int64_t kInterval = 2;
}  // namespace
bool CTCLossV2GradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  // Getting values

  blank_ = GetValue<int64_t>(primitive_->GetAttr("blank"));
  zero_infinity_ = GetValue<bool>(primitive_->GetAttr("zero_infinity"));

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return true;
}

int CTCLossV2GradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto log_probs_shape = inputs[kIndex1]->GetShapeVector();
  time_series_ = log_probs_shape[kIndex0];
  batch_size_ = log_probs_shape[kIndex1];
  num_labels_ = log_probs_shape[kIndex2];
  const auto target_shape = inputs[kIndex2]->GetShapeVector();
  max_target_length_ = target_shape[kIndex1];

  log_probs_shape_.x = LongToSize(time_series_);
  log_probs_shape_.y = LongToSize(batch_size_);
  log_probs_shape_.z = LongToSize(num_labels_);

  log_alpha_shape_.x = LongToSize(batch_size_);
  log_alpha_shape_.y = LongToSize(time_series_);
  log_alpha_shape_.z = LongToSize(kInterval * max_target_length_ + 1);

  const size_t scalar_type_size = abstract::TypeIdSize(inputs[kIndex0]->dtype_id());
  workspace_size_list_.clear();
  workspace_size_list_ = {
    LongToSize(batch_size_ * time_series_ * (kInterval * max_target_length_ + 1)) * scalar_type_size,
  };
  return KRET_OK;
}

template <typename scalar_t, typename target_t>
bool CTCLossV2GradGpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                             const std::vector<kernel::KernelTensor *> &workspace,
                                             const std::vector<kernel::KernelTensor *> &outputs) {
  auto grad_out = reinterpret_cast<scalar_t *>(inputs[kIndex0]->device_ptr());
  auto log_probs = reinterpret_cast<scalar_t *>(inputs[kIndex1]->device_ptr());
  auto targets = reinterpret_cast<target_t *>(inputs[kIndex2]->device_ptr());
  auto input_lengths = reinterpret_cast<target_t *>(inputs[kIndex3]->device_ptr());
  auto target_lengths = reinterpret_cast<target_t *>(inputs[kIndex4]->device_ptr());
  auto neg_log_likelihood = reinterpret_cast<scalar_t *>(inputs[kIndex5]->device_ptr());
  auto log_alpha = reinterpret_cast<scalar_t *>(inputs[kIndex6]->device_ptr());

  auto log_beta = reinterpret_cast<scalar_t *>(workspace[kIndex0]->device_ptr());
  auto grad = reinterpret_cast<scalar_t *>(outputs[kIndex0]->device_ptr());

  CalCTCLossGradV2<scalar_t, target_t>(grad_out, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood,
                                       log_alpha, log_beta, batch_size_, time_series_, num_labels_, max_target_length_,
                                       zero_infinity_, blank_, log_probs_shape_, log_alpha_shape_, grad, device_id_,
                                       stream_ptr_);

  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &CTCLossV2GradGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, CTCLossV2GradGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &CTCLossV2GradGpuKernelMod::LaunchKernel<float, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &CTCLossV2GradGpuKernelMod::LaunchKernel<double, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &CTCLossV2GradGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &CTCLossV2GradGpuKernelMod::LaunchKernel<double, int64_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CTCLossV2Grad, CTCLossV2GradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

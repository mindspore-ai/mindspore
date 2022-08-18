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
#include "plugin/device/gpu/kernel/nn/ctcloss_v2_gpu_kernel.h"
#include <string>
#include <limits>
#include <memory>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/ctcloss_v2_impl.cuh"
#include "mindspore/core/ops/ctc_loss_v2.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
using KernelRunFunc = CTCLossV2GpuKernelMod::KernelRunFunc;
constexpr auto kInterval = 2;
bool CTCLossV2GpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }

  // Getting values
  auto kernel_ptr = std::make_shared<ops::CTCLossV2>(base_operator->GetPrim());
  blank_ = kernel_ptr->get_blank();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}
int CTCLossV2GpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  const auto log_probs_shape = inputs[kIndex0]->GetShapeVector();
  const auto target_shape = inputs[kIndex1]->GetShapeVector();
  const auto input_length_shape = inputs[kIndex2]->GetShapeVector();
  const auto target_length_shape = inputs[kIndex3]->GetShapeVector();

  is_null_input_ = CHECK_NULL_INPUT(log_probs_shape) || CHECK_NULL_INPUT(target_shape) ||
                   CHECK_NULL_INPUT(input_length_shape) || CHECK_NULL_INPUT(target_length_shape);
  if (is_null_input_) {
    return KRET_OK;
  }

  time_series_ = log_probs_shape[kIndex0];
  batch_sizes_ = log_probs_shape[kIndex1];
  num_labels_ = log_probs_shape[kIndex2];

  max_target_length_ = target_shape[kIndex1];

  if (!(blank_ >= 0 && blank_ < num_labels_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << ", the attr blank must be in label range [ 0, " << num_labels_
                  << " ), but got value " << blank_ << ".";
    return KRET_RESIZE_FAILED;
  }
  if (input_length_shape.size() != 1 || input_length_shape[0] != batch_sizes_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'input_length' must be one-dimensional, "
                     "and the size is equal to batch_size: "
                  << batch_sizes_ << ", but got the shape of 'input_length': " << Vector2Str(input_length_shape) << ".";
    return KRET_RESIZE_FAILED;
  }
  if (target_length_shape.size() != 1 || target_length_shape[0] != batch_sizes_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'target_length' must be one-dimensional, "
                     "and the size is equal to batch_size: "
                  << batch_sizes_ << ", but got the shape of 'target_length': " << Vector2Str(target_length_shape)
                  << ".";
    return KRET_RESIZE_FAILED;
  }

  log_probs_shape_.x = LongToSize(time_series_);
  log_probs_shape_.y = LongToSize(batch_sizes_);
  log_probs_shape_.z = LongToSize(num_labels_);

  log_alpha_shape_.x = LongToSize(batch_sizes_);
  log_alpha_shape_.y = LongToSize(time_series_);
  log_alpha_shape_.z = LongToSize(kInterval * max_target_length_ + 1);

  return KRET_OK;
}

template <typename S, typename T>
bool CTCLossV2GpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
  auto log_probs_p = GetDeviceAddress<S>(inputs, kIndex0);
  auto target_p = GetDeviceAddress<T>(inputs, kIndex1);
  auto input_len_p = GetDeviceAddress<T>(inputs, kIndex2);
  auto target_len_p = GetDeviceAddress<T>(inputs, kIndex3);

  auto neg_log_p = GetDeviceAddress<S>(outputs, kIndex0);
  auto log_alpha_p = GetDeviceAddress<S>(outputs, kIndex1);

  CalCTCLossV2<S, T>(log_probs_p, target_p, input_len_p, target_len_p, batch_sizes_, max_target_length_, time_series_,
                     blank_, log_probs_shape_, log_alpha_shape_, neg_log_p, log_alpha_p, device_id_, stream_ptr_);

  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &CTCLossV2GpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &CTCLossV2GpuKernelMod::LaunchKernel<float, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &CTCLossV2GpuKernelMod::LaunchKernel<double, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &CTCLossV2GpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &CTCLossV2GpuKernelMod::LaunchKernel<double, int64_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CTCLossV2, CTCLossV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

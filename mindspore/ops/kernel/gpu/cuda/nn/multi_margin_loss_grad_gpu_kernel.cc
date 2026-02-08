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

#include "kernel/gpu/cuda/nn/multi_margin_loss_grad_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "include/curand.h"
#include "ir/dtype/empty.h"
#include "mindspore/ops/infer/grad/multi_margin_loss_grad.h"
#include "kernel/gpu/cuda_impl/cuda_ops/multi_margin_loss_grad_impl.cuh"

namespace mindspore {
namespace kernel {
bool MultiMarginLossGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  p_ = GetValue<int64_t>(primitive_->GetAttr(ops::kP));
  if (p_ != p_num_1 && p_ != p_num_2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' p should be 1 or 2, but got " << p_;
    return false;
  }
  margin_ = GetValue<float>(primitive_->GetAttr(ops::kMargin));
  string reduction = GetValue<std::string>(primitive_->GetAttr(ops::kReduction));
  reduction_ = 1;
  if (reduction == "mean") {
    reduction_ = reduction_num_1;
  } else if (reduction == "sum") {
    reduction_ = reduction_num_0;
  } else if (reduction == "none") {
    reduction_ = reduction_num_2;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  return true;
}

int MultiMarginLossGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  input_elements_ = 0;
  output_size_list_.clear();
  workspace_size_list_.clear();
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  auto input_shape = inputs[kIndex1]->GetShapeVector();
  input_elements_ = SizeOf(input_shape);
  if (input_elements_ == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be greater than zero.";
    return KRET_RESIZE_FAILED;
  }
  nframe_ = input_shape.at(0);
  dim_ = input_shape.at(1);

  auto type = inputs[kIndex3]->GetType();
  has_weight_ = !type->isa<TypeNone>();
  size_t input_size = input_elements_ * unit_size_;
  output_size_list_.push_back(input_size);
  return KRET_OK;
}

template <typename T>
bool MultiMarginLossGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &workspace,
                                                   const std::vector<KernelTensor *> &outputs) {
  T *output_grad = GetDeviceAddress<T>(inputs, kIndex0);
  T *input = GetDeviceAddress<T>(inputs, kIndex1);
  int64_t *target = GetDeviceAddress<int64_t>(inputs, kIndex2);
  T *weight = nullptr;
  if (has_weight_) {
    weight = GetDeviceAddress<T>(inputs, kIndex3);
  }
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  auto status = MultiMarginLossGrad(p_, margin_, reduction_, nframe_, dim_, output_grad, input, target, weight, output,
                                    device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, MultiMarginLossGradGpuKernelMod::MultiMarginLossGradFunc>>
  MultiMarginLossGradGpuKernelMod::func_list_ = {{KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddOptionalInputAttr(kNumberTypeFloat16)
                                                    .AddOutputAttr(kNumberTypeFloat16),
                                                  &MultiMarginLossGradGpuKernelMod::LaunchKernel<half>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddOptionalInputAttr(kNumberTypeFloat64)
                                                    .AddOutputAttr(kNumberTypeFloat64),
                                                  &MultiMarginLossGradGpuKernelMod::LaunchKernel<double>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddOptionalInputAttr(kNumberTypeFloat32)
                                                    .AddOutputAttr(kNumberTypeFloat32),
                                                  &MultiMarginLossGradGpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> MultiMarginLossGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MultiMarginLossGradFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MultiMarginLossGrad, MultiMarginLossGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

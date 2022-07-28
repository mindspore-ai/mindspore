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

#include "plugin/device/gpu/kernel/nn/multi_margin_loss_grad_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "include/curand.h"
#include "mindspore/core/ops/grad/multi_margin_loss_grad.h"
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/multi_margin_loss_grad_impl.cuh"

namespace mindspore {
namespace kernel {
bool MultiMarginLossGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MultiMarginLossGrad>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' cast Cdist ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  p_ = kernel_ptr->get_p();
  if (p_ != p_num_1 && p_ != p_num_2) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' p should be 1 or 2, but got " << p_;
    return false;
  }
  margin_ = kernel_ptr->get_margin();
  string reduction = kernel_ptr->get_reduction();
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
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).first);
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  return true;
}

int MultiMarginLossGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &) {
  input_elements_ = 0;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  int inputs_size = 0;
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    inputs_size += 1;
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  auto input_shape = inputs.at(kIndex1)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  input_elements_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());
  if (input_elements_ == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be greater than zero.";
    return KRET_RESIZE_FAILED;
  }
  nframe_ = input_shape_.at(0);
  dim_ = input_shape_.at(1);
  if (inputs_size == has_weight_inputs_size) {
    has_weight_ = true;
  } else {
    has_weight_ = false;
  }
  size_t input_size = input_elements_ * unit_size_;
  input_size_list_.push_back(input_size);
  output_size_list_.push_back(input_size);
  return KRET_OK;
}

template <typename T>
bool MultiMarginLossGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &workspace,
                                                   const std::vector<AddressPtr> &outputs) {
  T *output_grad = GetDeviceAddress<T>(inputs, kIndex0);
  T *input = GetDeviceAddress<T>(inputs, kIndex1);
  int64_t *target = GetDeviceAddress<int64_t>(inputs, kIndex2);
  T *weight = nullptr;
  if (has_weight_) {
    weight = GetDeviceAddress<T>(inputs, kIndex3);
  }
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  MultiMarginLossGrad(p_, margin_, reduction_, nframe_, dim_, output_grad, input, target, weight, output, device_id_,
                      reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, MultiMarginLossGradGpuKernelMod::MultiMarginLossGradFunc>>
  MultiMarginLossGradGpuKernelMod::func_list_ = {{KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddOutputAttr(kNumberTypeFloat16),
                                                  &MultiMarginLossGradGpuKernelMod::LaunchKernel<half>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddOutputAttr(kNumberTypeFloat64),
                                                  &MultiMarginLossGradGpuKernelMod::LaunchKernel<double>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddOutputAttr(kNumberTypeFloat32),
                                                  &MultiMarginLossGradGpuKernelMod::LaunchKernel<float>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeFloat16)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddOutputAttr(kNumberTypeFloat16),
                                                  &MultiMarginLossGradGpuKernelMod::LaunchKernel<half>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeFloat64)
                                                    .AddInputAttr(kNumberTypeInt64)
                                                    .AddOutputAttr(kNumberTypeFloat64),
                                                  &MultiMarginLossGradGpuKernelMod::LaunchKernel<double>},
                                                 {KernelAttr()
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeFloat32)
                                                    .AddInputAttr(kNumberTypeInt64)
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

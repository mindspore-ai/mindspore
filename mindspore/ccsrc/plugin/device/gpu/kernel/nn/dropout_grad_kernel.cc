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
#include <algorithm>
#include <functional>
#include <map>
#include <utility>

#include "plugin/device/gpu/kernel/nn/dropout_grad_kernel.h"
#include "mindspore/core/ops/grad/dropout_grad.h"

namespace mindspore {
namespace kernel {
constexpr size_t kDropoutGradInputNum = 2;
constexpr size_t kDropoutGradOutputNum = 1;

bool DropoutGradBwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::DropoutGrad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast DropoutGrad ops failed!";
    return false;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDropoutGradInputNum, kernel_ptr->name());
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDropoutGradOutputNum, kernel_ptr->name());

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_ptr->name()
                      << "', it does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  keep_prob_ = kernel_ptr->get_keep_prob();
  return true;
}

int DropoutGradBwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &others) {
  ResetResource();
  dy_shape_ = inputs[kIndex0]->GetShapeVector();
  mask_shape_ = inputs[kIndex1]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  if (!(CHECK_SHAPE_POSITIVE(dy_shape_) && CHECK_SHAPE_POSITIVE(mask_shape_) && CHECK_SHAPE_POSITIVE(output_shape_))) {
    is_null_input_ = true;
    InitSizeLists();
    return 0;
  }

  MS_EXCEPTION_IF_CHECK_FAIL(!dy_shape_.empty(), "dy_shape_ should not be empty!");
  num_count_ = std::accumulate(dy_shape_.begin(), dy_shape_.end(), 1, std::multiplies<size_t>());

  dy_size_ = abstract::TypeIdSize(inputs[kIndex0]->GetDtype()) * num_count_;
  mask_size_ = abstract::TypeIdSize(inputs[kIndex1]->GetDtype()) * num_count_;
  output_size_ = abstract::TypeIdSize(outputs[kIndex0]->GetDtype()) * num_count_;

  InitSizeLists();
  return 0;
}

void DropoutGradBwdGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  dy_size_ = 0;
  mask_size_ = 0;
  output_size_ = 0;
  num_count_ = 0;
  input_size_list_.clear();
  output_size_list_.clear();
}

void DropoutGradBwdGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(dy_size_);
  input_size_list_.push_back(mask_size_);
  output_size_list_.push_back(output_size_);
}

template <typename T>
bool DropoutGradBwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *dy_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *mask_addr = GetDeviceAddress<T>(inputs, kIndex1);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);

  DropoutBackward(dy_addr, mask_addr, output_addr, num_count_, keep_prob_, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, DropoutGradBwdGpuKernelMod::DropoutGradFunc>> DropoutGradBwdGpuKernelMod::func_list_ =
  {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    &DropoutGradBwdGpuKernelMod::LaunchKernel<half>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    &DropoutGradBwdGpuKernelMod::LaunchKernel<float>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    &DropoutGradBwdGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> DropoutGradBwdGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DropoutGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, DropoutGrad, DropoutGradBwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

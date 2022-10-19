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

#include "plugin/device/gpu/kernel/math/cdist_gpu_kernel.h"
#include <utility>
#include <algorithm>

namespace mindspore {
namespace kernel {
constexpr size_t kOne = 1;
constexpr size_t kTwo = 2;
constexpr size_t kThree = 3;
bool CdistGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::Cdist>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float32, double], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  batch_ = 0;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  p_ = kernel_ptr_->get_p();
  return true;
}

int CdistGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> in_shape_x = inputs[0]->GetShapeVector();
  std::vector<int64_t> in_shape_y = inputs[1]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[0]->GetShapeVector();
  x_elements_ = std::accumulate(in_shape_x.begin(), in_shape_x.end(), 1, std::multiplies<int64_t>());
  y_elements_ = std::accumulate(in_shape_y.begin(), in_shape_y.end(), 1, std::multiplies<int64_t>());
  out_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  if (out_elements_ == 0) {
    is_null_input_ = true;
  }
  size_t in_shape_size = in_shape_x.size();
  if (in_shape_size != in_shape_y.size() || in_shape_size < kTwo) {
    MS_LOG(ERROR) << "invalid input shape, input_x shape size " << in_shape_size << ", input_y shape size "
                  << in_shape_y.size() << ", kernel_name_ " << kernel_name_;
    return KRET_RESIZE_FAILED;
  }
  for (size_t i = 0; i < in_shape_size - kTwo; i++) {
    batch_ += in_shape_x[i];
  }
  batch_ = (batch_ <= 0) ? 1 : batch_;
  x_row_ = in_shape_x[in_shape_size - kTwo];
  x_col_ = in_shape_x[in_shape_size - kOne];
  y_row_ = in_shape_y[in_shape_size - kTwo];

  size_t x_size = x_elements_ * unit_size_;
  size_t y_size = y_elements_ * unit_size_;
  size_t out_size = out_elements_ * unit_size_;
  input_size_list_.push_back(x_size);
  input_size_list_.push_back(y_size);
  output_size_list_.push_back(out_size);

  return KRET_OK;
}

template <typename T>
bool CdistGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs) {
  T *input_x = GetDeviceAddress<T>(inputs, 0);
  T *input_y = GetDeviceAddress<T>(inputs, 1);
  T *out_data = GetDeviceAddress<T>(outputs, 0);

  CalCdist(out_elements_, input_x, input_y, out_data, x_row_, y_row_, x_col_, p_, batch_, device_id_,
           reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, CdistGpuKernelMod::CdistFunc>> CdistGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &CdistGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &CdistGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> CdistGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CdistFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Cdist, CdistGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

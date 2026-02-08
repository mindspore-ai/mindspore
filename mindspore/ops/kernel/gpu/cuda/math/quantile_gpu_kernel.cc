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

#include "kernel/gpu/cuda/math/quantile_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "mindspore/ops/infer/quantile.h"
#include "kernel/gpu/cuda_impl/cuda_ops/quantile_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr int kQuantileDefaultDim = 10000;
bool QuantileGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  dim_ = GetValue<int64_t>(primitive_->GetAttr("dim"));
  ignorenan_ = GetValue<bool>(primitive_->GetAttr("ignore_nan"));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  input_unit_size_ = abstract::TypeIdSize(inputs[kIndex0]->dtype_id());
  q_unit_size_ = abstract::TypeIdSize(inputs[kIndex1]->dtype_id());
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  return true;
}

uint32_t MaybeWrapDim(int dim, int dim_post_expr) {
  if (dim == kQuantileDefaultDim) {
    return dim;
  }
  if (dim_post_expr <= 0) {
    dim_post_expr = 1;
  }
  int min = -dim_post_expr;
  int max = dim_post_expr - 1;
  if (dim < min || dim > max) {
    MS_LOG(ERROR) << "For Quantile, dimension out of range (expected to be in range of " << min << " and [ " << max
                  << "]).";
  }
  if (dim < 0) {
    dim += dim_post_expr;
  }
  return dim;
}

int QuantileGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
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
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  input_elements_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), size_t(1), std::multiplies<size_t>());
  if (input_elements_ == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be greater than zero.";
    return KRET_RESIZE_FAILED;
  }
  dim_ = MaybeWrapDim(dim_, input_shape.size());
  if (dim_ == kQuantileDefaultDim) {
    x_ = 1;
    y_ = 1;
    for (size_t i = 0; i < input_shape.size(); i++) y_ *= input_shape.at(i);
    z_ = 1;
    dim_ = 0;
  } else {
    x_ = 1;
    y_ = input_shape.at(dim_);
    z_ = 1;
    for (int i = 0; i < dim_; i++) x_ *= input_shape.at(i);
    for (size_t i = dim_ + 1; i < input_shape.size(); i++) z_ *= input_shape.at(i);
  }
  each_q_elements_ = input_elements_ / y_;
  size_t input_size = input_elements_ * input_unit_size_;
  output_size_list_.push_back(output_elements_ * input_unit_size_);
  ceil_power2_ = RoundUpPower2(y_);
  workspace_size_list_.push_back(input_size / y_ * ceil_power2_);
  workspace_size_list_.push_back(input_unit_size_);
  workspace_size_list_.push_back(output_elements_);
  return KRET_OK;
}

template <typename T>
bool QuantileGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  T *q = GetDeviceAddress<T>(inputs, kIndex1);
  T *out = GetDeviceAddress<T>(outputs, kIndex0);
  T *sort = GetDeviceAddress<T>(workspace, kIndex0);
  int *ret_flag_device = GetDeviceAddress<int>(workspace, kIndex1);
  int *nan_flags = GetDeviceAddress<int>(workspace, kIndex2);
  total_ = inputs[0]->size() / sizeof(T);
  if (total_ == 0) {
    MS_LOG(ERROR) << "For Quantile, input tensor must be non-empty";
  }
  int flag_in = 0;
  auto status =
    Quantile(input, q, out, sort, dim_, x_, y_, z_, each_q_elements_, output_elements_, &flag_in, ret_flag_device,
             nan_flags, ignorenan_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  if (flag_in == 1) {
    MS_EXCEPTION(ValueError) << "For Quantile, q out of range (expected to be in range of [0, 1]).";
    return false;
  }
  return true;
}

std::vector<std::pair<KernelAttr, QuantileGpuKernelMod::QuantileFunc>> QuantileGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &QuantileGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &QuantileGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> QuantileGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, QuantileFunc> &pair) { return pair.first; });
  return support_list;
}
}  // namespace kernel
}  // namespace mindspore

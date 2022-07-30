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

#include "plugin/device/gpu/kernel/math/complex_abs_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include <complex>
#include "include/curand.h"
#include "mindspore/core/ops/complex_abs.h"
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex_abs_impls.cuh"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
constexpr int MAX_DIMS = 7;
bool ComplexAbsGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  kernel_ptr_ = std::make_shared<ops::ComplexAbs>(base_operator->GetPrim());
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  out_unit_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).first);

  return true;
}

int ComplexAbsGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  auto x_long_shape = inputs.at(kIndex0)->GetShapeVector();
  std::vector<size_t> x_shape;
  (void)std::transform(x_long_shape.begin(), x_long_shape.end(), std::back_inserter(x_shape), LongToSize);
  if (x_shape.size() > MAX_DIMS) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", but got x: " << x_shape.size();
  }
  auto y_long_shape = outputs.at(kIndex0)->GetShapeVector();
  std::vector<size_t> y_shape;
  (void)std::transform(y_long_shape.begin(), y_long_shape.end(), std::back_inserter(y_shape), LongToSize);
  if (y_shape.size() > MAX_DIMS) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                      << ", but got y: " << y_shape.size();
  }
  x_shape_.resize(MAX_DIMS, 1);
  y_shape_.resize(MAX_DIMS, 1);

  for (size_t i = 0; i < x_shape.size(); i++) {
    x_count_ *= x_shape[i];
  }
  for (size_t i = 0; i < y_shape.size(); i++) {
    y_count_ *= y_shape[i];
  }
  size_t input_size = x_count_ * unit_size_;
  size_t output_size = y_count_ * out_unit_size_;
  input_size_list_.emplace_back(input_size);
  output_size_list_.emplace_back(output_size);
  return KRET_OK;
}

void ComplexAbsGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  x_count_ = 1;
  y_count_ = 1;
  x_shape_.clear();
  y_shape_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
}

template <typename T, typename S>
bool ComplexAbsGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs) {
  T *x = GetDeviceAddress<T>(inputs, kIndex0);
  S *y = GetDeviceAddress<S>(outputs, kIndex0);
  ComplexAbs(x_count_, x, y, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::vector<std::pair<KernelAttr, ComplexAbsGpuKernelMod::ComplexAbsFunc>> ComplexAbsGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
   &ComplexAbsGpuKernelMod::LaunchKernel<Complex<float>, float>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
   &ComplexAbsGpuKernelMod::LaunchKernel<Complex<double>, double>}};

std::vector<KernelAttr> ComplexAbsGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ComplexAbsFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ComplexAbs, ComplexAbsGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

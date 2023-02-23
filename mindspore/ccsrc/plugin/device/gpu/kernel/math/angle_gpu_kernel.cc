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

#include "plugin/device/gpu/kernel/math/angle_gpu_kernel.h"
#include <utility>
#include <functional>
#include <string>
#include <algorithm>
#include <memory>
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
void AngleGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

std::vector<KernelAttr> AngleGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, AngleFunc> &pair) { return pair.first; });
  return support_list;
}

bool AngleGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "'support complex64 or complex128, but got " << kernel_attr;
    return false;
  }
  input_dtype_ = inputs[0]->GetDtype();
  kernel_func_ = func_list_[index].second;
  return true;
}

int AngleGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  return ret;
}

template <typename T, typename S>
bool AngleGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input_ptr = GetDeviceAddress<T>(inputs, kIndex0);
  S *output_ptr = GetDeviceAddress<S>(outputs, kIndex0);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  output_size = outputs[0]->size / sizeof(S);
  auto status = CalAngle(output_size, input_ptr, output_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::vector<std::pair<KernelAttr, AngleGpuKernelMod::AngleFunc>> AngleGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
   &AngleGpuKernelMod::LaunchKernel<Complex<float>, float>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
   &AngleGpuKernelMod::LaunchKernel<Complex<double>, double>}};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Angle, AngleGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

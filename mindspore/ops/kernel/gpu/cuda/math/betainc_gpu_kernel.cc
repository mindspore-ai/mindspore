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

#include "kernel/gpu/cuda/math/betainc_gpu_kernel.h"
#include <utility>
#include <algorithm>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/betainc_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBetaincInputsNum = 3;
constexpr size_t kAIndex = 0;
constexpr size_t kBIndex = 1;
constexpr size_t kXIndex = 2;
}  // namespace

bool BetaincGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != kBetaincInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs should be 3, but got " << inputs.size();
    return false;
  }
  constexpr int OUTPUT_NUM = 1;
  if (outputs.size() != OUTPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << outputs.size();
    return false;
  }
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float32, float64], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}
int BetaincGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  input_element_ = SizeOf(inputs[0]->GetShapeVector());
  return ret;
}

bool BetaincGpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                 const std::vector<kernel::KernelTensor *> &workspace,
                                 const std::vector<kernel::KernelTensor *> &outputs, void *cuda_stream) {
  return kernel_func_(this, inputs, outputs, workspace, cuda_stream);
}

template <typename T>
bool BetaincGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs,
                                       const std::vector<KernelTensor *> &workspace, void *cuda_stream) {
  T *input_a = GetDeviceAddress<T>(inputs, kAIndex);
  T *input_b = GetDeviceAddress<T>(inputs, kBIndex);
  T *input_x = GetDeviceAddress<T>(inputs, kXIndex);
  T *output = GetDeviceAddress<T>(outputs, kAIndex);
  auto status = CalBetainc(input_element_, input_a, input_b, input_x, output, device_id_,
                           reinterpret_cast<cudaStream_t>(cuda_stream));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, BetaincGpuKernelMod::KernelFunc>> BetaincGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &BetaincGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &BetaincGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> BetaincGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Betainc, BetaincGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

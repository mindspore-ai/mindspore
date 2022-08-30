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

#include "plugin/device/gpu/kernel/math/cauchy_gpu_kernel.h"
#include <curand_kernel.h>
#include <map>
#include <vector>
#include <algorithm>
#include <utility>
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "mindspore/core/ops/cauchy.h"
#include "include/common/utils/convert_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cauchy_impl.cuh"

namespace mindspore {
namespace kernel {
std::vector<std::pair<KernelAttr, CauchyGpuKernelMod::CauchyFunc>> CauchyGpuKernelMod::func_list_ = {
  {KernelAttr().AddOutputAttr(kNumberTypeFloat32), &CauchyGpuKernelMod::LaunchKernel<float>}};

bool CauchyGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  size_ = GetValue<std::vector<int64_t>>(base_operator->GetAttr("size"));
  median_ = GetValue<float>(base_operator->GetAttr("median"));
  sigma_ = GetValue<float>(base_operator->GetAttr("sigma"));

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'Cauchy', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int CauchyGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  output_elements_ = std::accumulate(size_.begin(), size_.end(), int64_t(1), std::multiplies{});

  workspace_size_list_.clear();
  workspace_size_list_ = {
    output_elements_ * sizeof(float),
  };

  return KRET_OK;
}

template <typename T>
bool CauchyGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                      const std::vector<AddressPtr> &outputs) {
  auto output_addr = GetDeviceAddress<T>(outputs, kDim0);
  auto seed = static_cast<uint64_t>(time(NULL));
  Cauchy(output_addr, seed, median_, sigma_, output_elements_, device_id_,
         reinterpret_cast<cudaStream_t>(cuda_stream_));

  return true;
}

std::vector<KernelAttr> CauchyGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CauchyGpuKernelMod::CauchyFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Cauchy, CauchyGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

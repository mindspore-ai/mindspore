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

#include "plugin/device/cpu/kernel/cauchy_cpu_kernel.h"
#include <vector>
#include <cmath>
#include <type_traits>
#include <memory>
#include <functional>
#include <random>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/arithmetic_cpu_kernel.h"

namespace mindspore {
namespace kernel {
const size_t kCauchyOutputNum = 1;

// namespace

void CauchyCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kCauchyOutputNum, common::AnfAlgo::GetCNodeName(kernel_node));

  std::vector<int64_t> size_ = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "size");
  sigma_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "sigma");
  median_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "median");
  auto y_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  for (size_t i = 0; i < size_.size(); i++) {
    if (size_[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For Cauchy, each dimension of size must be greater than zero.";
    }
    if (size_[i] != y_shape[i]) {
      MS_EXCEPTION(ValueError) << "For Cauchy, output shape not equal with size in dimension " << i << " .";
    }
  }
}
bool CauchyCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &,
                                const std::vector<kernel::AddressPtr> &outputs) {
  LaunchKernel<float>(outputs);
  return true;
}

template <typename T>
bool CauchyCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &outputs) {
  T *y_data = reinterpret_cast<T *>(outputs[0]->addr);
  std::random_device rd;
  std::default_random_engine generator(rd());
  std::cauchy_distribution<float> cauchy_d(median_, sigma_);
  auto end = outputs[0]->size / sizeof(T);

  for (size_t i = 0; i < end; ++i) {
    float data = cauchy_d(generator);
    y_data[i] = static_cast<T>(data);
  }

  return true;
}

std::vector<KernelAttr> CauchyCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr().AddOutputAttr(kNumberTypeFloat16),
                                                 KernelAttr().AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Cauchy, CauchyCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DATA_FORMAT_VEC_PERMUTE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DATA_FORMAT_VEC_PERMUTE_CPU_KERNEL_H_

#include <string>
#include <utility>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class DataFormatVecPermuteCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  DataFormatVecPermuteCpuKernelMod() = default;

  ~DataFormatVecPermuteCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using DataFormatVecPermuteFunc =
    std::function<bool(DataFormatVecPermuteCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, DataFormatVecPermuteFunc>> func_list_;
  DataFormatVecPermuteFunc kernel_func_;
  std::string src_format_;
  std::string dst_format_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  TypeId input_type_{kTypeUnknown};
  TypeId output_type_{kTypeUnknown};
  size_t dim_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DATA_FORMAT_VEC_PERMUTE_CPU_KERNEL_H_

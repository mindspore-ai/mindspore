/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_BINCOUNT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_BINCOUNT_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "utils/ms_utils.h"
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
class BincountCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  BincountCpuKernelMod() = default;
  ~BincountCpuKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::vector<int64_t> input_arr_sizes_;
  std::vector<int64_t> input_weights_sizes_;
  std::vector<int64_t> input_size_sizes_;
  std::vector<int64_t> output_sizes_;
  TypeId dt_arr_{kTypeUnknown};
  TypeId dt_weights_{kTypeUnknown};
  void SetMap();
  std::map<int, std::map<int, std::function<void(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                                 const std::vector<AddressPtr> &, std::vector<int64_t> &, int32_t,
                                                 const std::vector<int64_t> &, const std::vector<int64_t> &)>>>
    calls_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_BINCOUNT_CPU_KERNEL_H_

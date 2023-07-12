/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_AKG_AKG_CPU_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_AKG_AKG_CPU_KERNEL_MOD_H_
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "kernel/kernel.h"
#include "plugin/device/cpu/kernel/akg/akg_kernel_loader.h"
#include "plugin/device/cpu/kernel/akg/akg_cpu_kernel_manager.h"
#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"

namespace mindspore {
namespace kernel {
class AkgCpuKernelManager : public AkgCpuKernelManagerAbs {
 public:
  AkgCpuKernelManager() = default;
  virtual ~AkgCpuKernelManager();

  void *GetFunction(const std::string &kernel_name) override;
  void GetFunctionAndKernelName(const std::string &fn, const std::string &kernel_name, std::string *fn_so,
                                std::string *fn_kernel) const override;
  AkgLibraryLoader object_loader;
};
using AkgCpuKernelManagerPtr = std::shared_ptr<AkgCpuKernelManager>;
class AkgCpuKernelMod : public CpuKernelMod {
 public:
  explicit AkgCpuKernelMod(const KernelPackPtr &kp);
  ~AkgCpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *) override;

  std::vector<KernelAttr> GetOpSupport() { return {}; }

  static AkgCpuKernelManagerPtr kernel_manager_;

 private:
  void *launch_func_;
};

using AkgCpuKernelModPtr = std::shared_ptr<AkgCpuKernelMod>;
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_AKG_AKG_CPU_KERNEL_MOD_H_

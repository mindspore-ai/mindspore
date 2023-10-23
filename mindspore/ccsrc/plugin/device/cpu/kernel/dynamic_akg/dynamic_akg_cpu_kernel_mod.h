/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_DYNAMIC_AKG_CPU_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_DYNAMIC_AKG_CPU_KERNEL_MOD_H_
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include <unordered_map>
#include "plugin/device/cpu/kernel/akg/akg_cpu_kernel_manager.h"
#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"

namespace mindspore {
namespace kernel {
class DynamicAkgCpuKernelManager : public AkgCpuKernelManagerAbs {
 public:
  DynamicAkgCpuKernelManager() = default;
  ~DynamicAkgCpuKernelManager();

  void *GetFunction(const std::string &kernel_name) override;
  void GetFunctionAndKernelName(const std::string &fn, const std::string &kernel_name, std::string *fn_so,
                                std::string *fn_kernel) const override;
};
using DynamicAkgCpuKernelManagerPtr = std::shared_ptr<DynamicAkgCpuKernelManager>;
class DynamicAkgCpuKernelMod : public CpuKernelMod {
 public:
  explicit DynamicAkgCpuKernelMod(const std::string &kernel_name);
  explicit DynamicAkgCpuKernelMod(const KernelPackPtr &kernel_pack);
  ~DynamicAkgCpuKernelMod() = default;

  bool Init(const BaseOperatorPtr & /* base_operator */, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;
  void SetKernelDynamicStatus(bool is_dynamic) { is_dynamic_ = is_dynamic; }

  enum KernelModType GetKernelModType() const override { return KernelModType::DynamicAkgCpuKernelMod; }

  std::vector<KernelAttr> GetOpSupport() override { return {}; }

  static DynamicAkgCpuKernelManagerPtr kernel_manager_;

 private:
  void *launch_func_{nullptr};
  bool is_dynamic_{false};

  std::vector<std::vector<int64_t>> shape_list_;
  std::vector<size_t> ndims_;
};

using DynamicAkgCpuKernelModPtr = std::shared_ptr<DynamicAkgCpuKernelMod>;
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_DYNAMIC_AKG_CPU_KERNEL_MOD_H_

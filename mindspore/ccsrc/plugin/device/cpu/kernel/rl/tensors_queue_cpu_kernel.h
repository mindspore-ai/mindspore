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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_TENSORS_QUEUE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_TENSORS_QUEUE_CPU_KERNEL_H_

#include "plugin/device/cpu/kernel/rl/tensors_queue_cpu_base.h"
#include <string>
#include <vector>
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class TensorsQueueCreateCpuKernelMod : public TensorsQueueCPUBaseMod {
 public:
  TensorsQueueCreateCpuKernelMod();
  ~TensorsQueueCreateCpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;
  void InitKernel(const CNodePtr &kernel_node) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr().AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

 private:
  std::string name_;
  int64_t size_;
  int64_t elements_num_;
  std::vector<std::vector<int64_t>> shapes_;
  TypePtr type_;
};

class TensorsQueuePutCpuKernelMod : public TensorsQueueCPUBaseMod {
 public:
  TensorsQueuePutCpuKernelMod();
  ~TensorsQueuePutCpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;
  void InitKernel(const CNodePtr &kernel_node) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
    return support_list;
  }

 private:
  int64_t elements_num_ = 0;
  TypeId type_ = kTypeUnknown;
};

class TensorsQueueGetCpuKernelMod : public TensorsQueueCPUBaseMod {
 public:
  TensorsQueueGetCpuKernelMod();
  ~TensorsQueueGetCpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;
  void InitKernel(const CNodePtr &kernel_node) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
    return support_list;
  }

 private:
  int64_t elements_num_;
  bool pop_after_get_;
};

class TensorsQueueClearCpuKernelMod : public TensorsQueueCPUBaseMod {
 public:
  TensorsQueueClearCpuKernelMod();
  ~TensorsQueueClearCpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;
  void InitKernel(const CNodePtr &kernel_node) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }
};

class TensorsQueueCloseCpuKernelMod : public TensorsQueueCPUBaseMod {
 public:
  TensorsQueueCloseCpuKernelMod();
  ~TensorsQueueCloseCpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;
  void InitKernel(const CNodePtr &kernel_node) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }
};

class TensorsQueueSizeCpuKernelMod : public TensorsQueueCPUBaseMod {
 public:
  TensorsQueueSizeCpuKernelMod();
  ~TensorsQueueSizeCpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;
  void InitKernel(const CNodePtr &kernel_node) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_TENSORS_QUEUE_CPU_KERNEL_H_

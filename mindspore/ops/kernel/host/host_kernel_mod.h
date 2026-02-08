/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_OPS_KERNEL_HOST_KERNEL_MOD_H_
#define MINDSPORE_OPS_KERNEL_HOST_KERNEL_MOD_H_
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <utility>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/host/visible.h"

namespace mindspore {
namespace kernel {

class OPS_HOST_EXPORT HostKernelMod : public KernelMod {
 public:
  HostKernelMod() = default;
  ~HostKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return true;
  }

  virtual void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  }

  enum KernelModType GetKernelModType() const override { return KernelModType::HostKernelMod; }

  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not supported in host kernel module.";
  }
};

using HostKernelModPtr = std::shared_ptr<HostKernelMod>;
using HostKernelModPtrList = std::vector<HostKernelModPtr>;
using HostKernelCreater = std::function<std::shared_ptr<HostKernelMod>()>;

class OPS_HOST_EXPORT HostKernelFactory {
  HostKernelFactory() = default;
  ~HostKernelFactory() = default;

 public:
  static HostKernelFactory &Get();
  void Register(const string &name, HostKernelCreater &&fun);
  static std::shared_ptr<HostKernelMod> Get(const string &name);

 private:
  std::map<string, HostKernelCreater> hostKernelMap_;
};

class OPS_HOST_EXPORT HostKernelRegister {
 public:
  HostKernelRegister(const string &name, HostKernelCreater &&fun) {
    HostKernelFactory::Get().Register(name, std::move(fun));
  }
  ~HostKernelRegister() = default;
};

#define MS_HOST_REG_KERNEL_REG(KNAME, clazz)                                                     \
  static_assert(std::is_base_of<HostKernelMod, clazz>::value, " must be base of HostKernelMod"); \
  static const HostKernelRegister g_##KNAME##_##_kernel_reg(#KNAME, []() {                       \
    std::shared_ptr<clazz> ptr = nullptr;                                                        \
    ptr = std::make_shared<clazz>();                                                             \
    MS_EXCEPTION_IF_NULL(ptr);                                                                   \
    return ptr;                                                                                  \
  });

#define MS_HOST_REG_KERNEL(KNAME, clazz) MS_HOST_REG_KERNEL_REG(KNAME, clazz)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_OPS_KERNEL_HOST_KERNEL_MOD_H_

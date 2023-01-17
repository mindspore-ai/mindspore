/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HOST_HOST_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HOST_HOST_KERNEL_MOD_H_
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <utility>
#include "plugin/device/ascend/kernel/ascend_kernel_mod.h"
namespace mindspore {
namespace kernel {
class HostKernelMod : public AscendKernelMod {
 public:
  HostKernelMod() = default;
  ~HostKernelMod() override = default;
  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              void *) override;
  std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &, uint32_t) override;
  bool Init(const AnfNodePtr &anf_node);
  int Resize(const BaseOperatorPtr &, const std::vector<KernelTensorPtr> &, const std::vector<KernelTensorPtr> &,
             const std::map<uint32_t, tensor::TensorPtr> &) override;
  enum KernelModType GetKernelModType() const override { return KernelModType::HostKernelMod; }

 protected:
  void SyncData() override{};
  std::string op_name_;
};

using HostKernelModPtr = std::shared_ptr<HostKernelMod>;
using HostKernelModPtrList = std::vector<HostKernelModPtr>;
using HostKernelCreater = std::function<std::shared_ptr<HostKernelMod>()>;

class HostKernelFactory {
  HostKernelFactory() = default;
  ~HostKernelFactory() = default;

 public:
  static HostKernelFactory &Get();
  void Register(const string &name, HostKernelCreater &&fun);
  static std::shared_ptr<HostKernelMod> Get(const string &name);

 private:
  std::map<string, HostKernelCreater> hostKernelMap_;
};

class HostKernelRegister {
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

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HOST_HOST_KERNEL_MOD_H_

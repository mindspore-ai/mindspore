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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_SIMU_SIMU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_SIMU_SIMU_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include "ir/anf.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
class SimuKernel : public KernelMod {
 public:
  SimuKernel() = default;
  ~SimuKernel() override = default;
  using KernelMod::Init;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    MS_LOG(INFO) << "SimuKernel default init with input size " << inputs.size() << " output size " << outputs.size();
    return true;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs, void *) override {
    MS_LOG(INFO) << "SimuKernel default launch with input size " << inputs.size() << " output size " << outputs.size();
    return true;
  }

  std::vector<KernelAttr> GetOpSupport() override {
    MS_LOG(EXCEPTION) << "This interface is not support in simu kernel module.";
  }
};

using SimuKernelCreater = std::function<std::shared_ptr<SimuKernel>()>;

class SimuKernelFactory {
  ~SimuKernelFactory() = default;

 public:
  static SimuKernelFactory &Get() {
    static SimuKernelFactory instance{};
    return instance;
  }

  void Register(const string &name, SimuKernelCreater &&fun) { kernel_map_[name] = fun; }

  static std::shared_ptr<SimuKernel> Get(const string &name) {
    auto &inst = Get();
    auto iter = inst.kernel_map_.find(name);
    if (iter != inst.kernel_map_.end() && iter->second != nullptr) {
      return iter->second();
    }
    return nullptr;
  }

 private:
  SimuKernelFactory() = default;
  std::map<string, SimuKernelCreater> kernel_map_;
};

class SimuKernelRegister {
 public:
  SimuKernelRegister(const string &name, SimuKernelCreater &&fun) {
    SimuKernelFactory::Get().Register(name, std::move(fun));
  }
  ~SimuKernelRegister() = default;
};

#define MS_SIMU_REG_KERNEL_REG(KNAME, clazz)                                               \
  static_assert(std::is_base_of<SimuKernel, clazz>::value, " must be base of SimuKernel"); \
  static const SimuKernelRegister g_##KNAME##_##_kernel_reg(#KNAME, []() {                 \
    std::shared_ptr<clazz> ptr = nullptr;                                                  \
    ptr = std::make_shared<clazz>();                                                       \
    MS_EXCEPTION_IF_NULL(ptr);                                                             \
    return ptr;                                                                            \
  });

#define MS_SIMU_REG_KERNEL(KNAME, clazz) MS_SIMU_REG_KERNEL_REG(KNAME, clazz)
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_SIMU_SIMU_KERNEL_H_

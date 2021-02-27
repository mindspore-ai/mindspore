/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_KERNEL_RUNTIME_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_KERNEL_RUNTIME_MANAGER_H_
#include <map>
#include <memory>
#include <string>
#include <functional>
#include <utility>
#include <mutex>
#include <unordered_set>
#include <vector>
#include "utils/ms_utils.h"
#include "runtime/device/kernel_runtime.h"
namespace mindspore {
namespace device {
using KernelRuntimeCreator = std::function<std::shared_ptr<KernelRuntime>()>;

class KernelRuntimeManager {
 public:
  static KernelRuntimeManager &Instance() {
    static KernelRuntimeManager instance;
    return instance;
  }
  void Register(const std::string &device_name, KernelRuntimeCreator &&runtime_creator);
  KernelRuntime *GetKernelRuntime(const std::string &device_name, uint32_t device_id);
  KernelRuntime *GetCurrentKernelRuntime();
  KernelRuntime *GetSingleKernelRuntime(const std::string &device_name, uint32_t device_id);
  void ReleaseKernelRuntime(const std::string &device_name, uint32_t device_id);
  void ClearRuntimeResource();
  void ClearGraphResource(uint32_t graph_id, const std::vector<AnfNodePtr> &inputs,
                          const std::unordered_set<ValueNodePtr> &value_nodes,
                          const std::vector<CNodePtr> &execution_order);

 private:
  KernelRuntimeManager() = default;
  ~KernelRuntimeManager() = default;
  DISABLE_COPY_AND_ASSIGN(KernelRuntimeManager);
  std::string GetDeviceKey(const std::string &device_name, uint32_t device_id);
  std::map<std::string, std::shared_ptr<KernelRuntime> > runtime_map_;
  std::map<std::string, KernelRuntimeCreator> runtime_creators_;
  std::mutex lock_;
};

class KernelRuntimeRegistrar {
 public:
  KernelRuntimeRegistrar(const std::string &device_name, KernelRuntimeCreator &&runtime_creator) {
    KernelRuntimeManager::Instance().Register(device_name, std::move(runtime_creator));
  }
  ~KernelRuntimeRegistrar() = default;
};

#define MS_REG_KERNEL_RUNTIME(DEVICE_NAME, RUNTIME_CLASS)                   \
  static const KernelRuntimeRegistrar g_kernel_runtime_##DEVICE_NAME##_reg( \
    DEVICE_NAME, []() { return std::make_shared<RUNTIME_CLASS>(); });
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_KERNEL_RUNTIME_MANAGER_H_

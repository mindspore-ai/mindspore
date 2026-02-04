/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_DUMP_ADAPTER_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_DUMP_ADAPTER_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "device_address/device_type.h"

namespace mindspore {
namespace dump {
constexpr auto kMSHookEnable = "MS_HOOK_ENABLE";
constexpr auto kEnable = "on";

class Adapter {
 public:
  Adapter() = default;

  virtual ~Adapter() {}

  virtual void AdaptOnStepBegin(uint32_t device_id, int step_count_num, std::vector<std::string> &&all_kernel_names,
                                bool is_kbyk) = 0;

  virtual void AdaptOnStepEnd() = 0;

  virtual void Load() = 0;
};

class AdapterManager {
 public:
  static AdapterManager &Instance();

  AdapterManager(const AdapterManager &) = delete;

  AdapterManager &operator=(const AdapterManager &) = delete;

  void RegisterAdapter(device::DeviceType backend, std::shared_ptr<Adapter> adapter_ptr);

  std::shared_ptr<Adapter> GetAdapterForBackend(device::DeviceType backend);

 private:
  AdapterManager() = default;

  std::map<device::DeviceType, std::shared_ptr<Adapter>> registered_adapters_;
};

#define REGISTER_ADAPTER(backend, type)                                                                 \
  namespace {                                                                                           \
  struct type##Register {                                                                               \
    type##Register() { AdapterManager::Instance().RegisterAdapter(backend, std::make_shared<type>()); } \
  };                                                                                                    \
  static type##Register global_##type##Register;                                                        \
  }

}  // namespace dump
}  // namespace mindspore

#endif

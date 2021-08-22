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

#ifndef MINDSPORE_LITE_SRC_REGISTRY_KERNEL_INTERFACE_REGISTRY_H_
#define MINDSPORE_LITE_SRC_REGISTRY_KERNEL_INTERFACE_REGISTRY_H_

#include <string>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include "include/registry/kernel_interface.h"
#include "include/model.h"

namespace mindspore {
namespace lite {
class KernelInterfaceRegistry {
 public:
  static KernelInterfaceRegistry *Instance() {
    static KernelInterfaceRegistry instance;
    return &instance;
  }

  std::shared_ptr<kernel::KernelInterface> GetKernelInterface(const std::string &provider,
                                                              const schema::Primitive *primitive);
  int CustomReg(const std::string &provider, const std::string &op_type, kernel::KernelInterfaceCreator creator);
  int Reg(const std::string &provider, int op_type, kernel::KernelInterfaceCreator creator);
  virtual ~KernelInterfaceRegistry();

 private:
  KernelInterfaceRegistry() = default;
  std::shared_ptr<kernel::KernelInterface> GetCacheInterface(const std::string &provider, int op_type);
  std::shared_ptr<kernel::KernelInterface> GetCustomCacheInterface(const std::string &provider,
                                                                   const std::string &type);
  std::shared_ptr<kernel::KernelInterface> GetCustomKernelInterface(const schema::Primitive *primitive);

  std::mutex mutex_;
  // key: provider
  std::map<std::string, kernel::KernelInterfaceCreator *> kernel_creators_;
  std::map<std::string, std::map<int, std::shared_ptr<kernel::KernelInterface>>> kernel_interfaces_;
  // key: provider        key: custom type
  std::map<std::string, std::map<std::string, kernel::KernelInterfaceCreator>> custom_creators_;
  std::map<std::string, std::map<std::string, std::shared_ptr<kernel::KernelInterface>>> custom_kernels_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_REGISTRY_KERNEL_INTERFACE_REGISTRY_H_

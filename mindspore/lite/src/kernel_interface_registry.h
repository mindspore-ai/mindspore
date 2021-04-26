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

#ifndef MINDSPORE_LITE_SRC_KERNEL_DEV_DELEGATE_REGISTRY_H_
#define MINDSPORE_LITE_SRC_KERNEL_DEV_DELEGATE_REGISTRY_H_

#include <string>
#include <unordered_map>
#include "src/kernel_interface.h"

namespace mindspore {
namespace lite {
class KernelInterfaceRegistry {
 public:
  static KernelInterfaceRegistry *Instance() {
    static KernelInterfaceRegistry instance;
    return &instance;
  }

  int Reg(const std::string &vendor, const int &op_type, kernel::KernelInterfaceCreator creator);
  virtual ~KernelInterfaceRegistry() = default;

 private:
  KernelInterfaceRegistry() = default;

  std::unordered_map<size_t, kernel::KernelInterfaceCreator *> kernel_interfaces_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_KERNEL_DEV_DELEGATE_REGISTRY_H_

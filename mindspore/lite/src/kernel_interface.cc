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
#include "src/kernel_interface.h"
#include "src/kernel_interface_registry.h"

namespace mindspore {
namespace kernel {
RegisterKernelInterface *RegisterKernelInterface::Instance() {
  static RegisterKernelInterface instance;
  return &instance;
}

int RegisterKernelInterface::Reg(const std::string &vendor, const int op_type, KernelInterfaceCreator creator) {
  return lite::KernelInterfaceRegistry::Instance()->Reg(vendor, op_type, creator);
}
}  // namespace kernel
}  // namespace mindspore

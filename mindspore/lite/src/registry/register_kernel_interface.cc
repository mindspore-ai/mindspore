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
#include "include/registry/register_kernel_interface.h"
#include <set>
#include <utility>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/registry/kernel_interface_registry.h"

namespace mindspore {
namespace registry {
Status RegisterKernelInterface::Reg(const std::vector<char> &provider, int op_type,
                                    const KernelInterfaceCreator creator) {
  return KernelInterfaceRegistry::Instance()->Reg(CharToString(provider), op_type, creator);
}

Status RegisterKernelInterface::CustomReg(const std::vector<char> &provider, const std::vector<char> &op_type,
                                          const KernelInterfaceCreator creator) {
  return KernelInterfaceRegistry::Instance()->CustomReg(CharToString(provider), CharToString(op_type), creator);
}

std::shared_ptr<kernel::KernelInterface> RegisterKernelInterface::GetKernelInterface(const std::vector<char> &provider,
                                                                                     const schema::Primitive *primitive,
                                                                                     const kernel::Kernel *kernel) {
  return KernelInterfaceRegistry::Instance()->GetKernelInterface(CharToString(provider), primitive, kernel);
}
}  // namespace registry
}  // namespace mindspore

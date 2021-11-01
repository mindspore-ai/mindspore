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
Status RegisterKernelInterface::Reg(const std::string &provider, int op_type, const KernelInterfaceCreator creator) {
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
  return KernelInterfaceRegistry::Instance()->Reg(provider, op_type, creator);
#else
  MS_LOG(ERROR) << unsupport_custom_kernel_register_log;
  return kLiteNotSupport;
#endif
}

Status RegisterKernelInterface::CustomReg(const std::string &provider, const std::string &op_type,
                                          const KernelInterfaceCreator creator) {
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
  return KernelInterfaceRegistry::Instance()->CustomReg(provider, op_type, creator);
#else
  MS_LOG(ERROR) << unsupport_custom_kernel_register_log;
  return kLiteNotSupport;
#endif
}

std::shared_ptr<kernel::KernelInterface> RegisterKernelInterface::GetKernelInterface(const std::string &provider,
                                                                                     const schema::Primitive *primitive,
                                                                                     const kernel::Kernel *kernel) {
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
  return KernelInterfaceRegistry::Instance()->GetKernelInterface(provider, primitive, kernel);
#else
  MS_LOG(ERROR) << unsupport_custom_kernel_register_log;
  return nullptr;
#endif
}
}  // namespace registry
}  // namespace mindspore

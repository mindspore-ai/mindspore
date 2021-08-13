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

#include "include/registry/register_kernel.h"
#include <set>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/registry/register_kernel_impl.h"

namespace mindspore {
namespace registry {
Status RegisterKernel::RegCustomKernel(const std::string &arch, const std::string &provider, DataType data_type,
                                       const std::string &type, CreateKernel creator) {
#ifdef ENABLE_CUSTOM_KERNEL_REGISTRY
  return RegistryKernelImpl::GetInstance()->RegCustomKernel(arch, provider, data_type, type, creator);
#else
  MS_LOG(ERROR) << unsupport_custom_kernel_register_log;
  return lite::RET_NOT_SUPPORT;
#endif
}

Status RegisterKernel::RegKernel(const std::string &arch, const std::string &provider, DataType data_type, int op_type,
                                 CreateKernel creator) {
#ifdef ENABLE_CUSTOM_KERNEL_REGISTRY
  return RegistryKernelImpl::GetInstance()->RegKernel(arch, provider, data_type, op_type, creator);
#else
  MS_LOG(ERROR) << unsupport_custom_kernel_register_log;
  return lite::RET_NOT_SUPPORT;
#endif
}

CreateKernel RegisterKernel::GetCreator(const schema::Primitive *primitive, KernelDesc *desc) {
#ifdef ENABLE_CUSTOM_KERNEL_REGISTRY
  return RegistryKernelImpl::GetInstance()->GetProviderCreator(primitive, desc);
#else
  MS_LOG(ERROR) << unsuppor_custom_kernel_register_log;
  return lite::RET_NOT_SUPPORT;
#endif
}
}  // namespace registry
}  // namespace mindspore

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
Status RegisterKernel::RegCustomKernel(const std::vector<char> &arch, const std::vector<char> &provider,
                                       DataType data_type, const std::vector<char> &type, const CreateKernel creator) {
  return RegistryKernelImpl::GetInstance()->RegCustomKernel(CharToString(arch), CharToString(provider), data_type,
                                                            CharToString(type), creator);
}

Status RegisterKernel::RegKernel(const std::vector<char> &arch, const std::vector<char> &provider, DataType data_type,
                                 int op_type, const CreateKernel creator) {
  return RegistryKernelImpl::GetInstance()->RegKernel(CharToString(arch), CharToString(provider), data_type, op_type,
                                                      creator);
}

CreateKernel RegisterKernel::GetCreator(const schema::Primitive *primitive, KernelDescHelper *desc) {
  if (desc == nullptr || primitive == nullptr) {
    return nullptr;
  }
  KernelDesc kernel_desc = {desc->data_type, desc->type, CharToString(desc->arch), CharToString(desc->provider)};
  auto ret = RegistryKernelImpl::GetInstance()->GetProviderCreator(primitive, &kernel_desc);
  desc->arch = StringToChar(kernel_desc.arch);
  return ret;
}
}  // namespace registry
}  // namespace mindspore

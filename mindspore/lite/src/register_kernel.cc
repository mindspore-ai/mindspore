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
#include "src/register_kernel.h"
#include "src/kernel_registry.h"

namespace mindspore {
namespace kernel {
RegisterKernel *RegisterKernel::GetInstance() {
  static RegisterKernel instance;
  return &instance;
}

int RegisterKernel::RegKernel(const std::string &arch, const std::string &vendor, const TypeId data_type,
                              const int op_type, CreateKernel creator) {
  return lite::KernelRegistry::GetInstance()->RegKernel(arch, vendor, data_type, op_type, creator);
}
}  // namespace kernel
}  // namespace mindspore

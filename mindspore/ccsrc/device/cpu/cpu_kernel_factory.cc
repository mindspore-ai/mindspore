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

#include "device/cpu/cpu_kernel_factory.h"

#include <memory>
#include <iostream>
#include <string>

namespace mindspore {
namespace device {
namespace cpu {
CPUKernelFactory &CPUKernelFactory::Get() {
  static CPUKernelFactory instance;
  return instance;
}

void CPUKernelFactory::Register(const std::string &kernel_name, CPUKernelCreator &&kernel_creator) {
  if (kernel_creators_.find(kernel_name) == kernel_creators_.end()) {
    (void)kernel_creators_.emplace(kernel_name, kernel_creator);
    MS_LOG(DEBUG) << "CPUKernelFactory register operator: " << kernel_name;
  }
}

std::shared_ptr<CPUKernel> CPUKernelFactory::Create(const std::string &kernel_name) {
  auto iter = kernel_creators_.find(kernel_name);
  if (iter != kernel_creators_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    return (iter->second)();
  }
  return nullptr;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

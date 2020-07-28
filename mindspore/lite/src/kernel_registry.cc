/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/kernel_registry.h"

using mindspore::kernel::KernelCreator;
using mindspore::kernel::KernelKey;
using mindspore::kernel::KERNEL_ARCH;

namespace mindspore::lite {
KernelRegistry::KernelRegistry() {}

KernelRegistry::~KernelRegistry() {}

KernelRegistry *KernelRegistry::GetInstance() {
  static KernelRegistry instance;
  return &instance;
}

KernelCreator KernelRegistry::GetKernelCreator(const KernelKey &desc) {
  auto it = creators.find(desc);
  if (it != creators.end()) {
    return it->second;
  }

  // if not find, use cpu kernel
  KernelKey cpuDesc {kernel::KERNEL_ARCH::kCPU, desc.type};
  it = creators.find(cpuDesc);
  if (it != creators.end()) {
    return it->second;
  }
  return nullptr;
}

void KernelRegistry::RegKernel(const KernelKey desc, KernelCreator creator) { creators[desc] = creator; }

void KernelRegistry::RegKernel(const KERNEL_ARCH arch, const schema::PrimitiveType type, KernelCreator creator) {
  KernelKey desc = {arch, type};
  creators[desc] = creator;
}

bool KernelRegistry::Merge(const std::unordered_map<KernelKey, KernelCreator> &newCreators) { return false; }

const std::map<KernelKey, KernelCreator> &KernelRegistry::GetKernelCreators() { return creators; }
}  // namespace mindspore::lite


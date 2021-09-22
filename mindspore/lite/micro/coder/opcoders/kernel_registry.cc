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

#include "coder/opcoders/kernel_registry.h"
#include <set>
#include "schema/ops_generated.h"

namespace mindspore::lite::micro {
KernelRegistry *KernelRegistry::GetInstance() {
  static KernelRegistry reg;
  return &reg;
}

void KernelRegistry::RegisterKernel(schema::PrimitiveType op) { registry_.insert(op); }

bool KernelRegistry::CheckRegistered(schema::PrimitiveType op) { return registry_.find(op) != registry_.end(); }

bool KernelRegistry::HasKernelRegistered() { return !registry_.empty(); }

std::string KernelRegistry::GenKernelInterface(const char *func, const char *param) {
  return "int " + std::string(func) + "(TensorC *inputs, int input_num, TensorC *outputs, int output_num, " +
         std::string(param) + " *param);";
}
}  // namespace mindspore::lite::micro

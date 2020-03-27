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

#include "src/op_factory.h"

namespace mindspore {
namespace predict {
OpFactory::OpFactory() { InitKernelManager(0, ""); }

OpFactory::~OpFactory() = default;

OpFactory *OpFactory::GetInstance() {
  static OpFactory instance;
  return &instance;
}

OpBase *OpFactory::GetOp(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, const OpDef &opDef,
                         const Context &ctx, const OpDesc &desc) {
  MS_ASSERT(GetRegistryInstance() != nullptr);
  auto *reg = GetRegistryInstance()->GetInstance<OpRegistry>(MODULE_REG_NAME_OP_REGISTRY);
  if (reg != nullptr) {
    auto creator = reg->GetOpCreator(desc);
    if (creator) {
      return creator(inputs, outputs, opDef, ctx, desc);
    }
  }
  MS_ASSERT(OpRegistry::GetInstance() != nullptr);
  auto creator = OpRegistry::GetInstance()->GetOpCreator(desc);
  if (creator) {
    return creator(inputs, outputs, opDef, ctx, desc);
  }
  return nullptr;
}
}  // namespace predict
}  // namespace mindspore

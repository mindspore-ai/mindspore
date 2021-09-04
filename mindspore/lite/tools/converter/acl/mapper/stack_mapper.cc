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

#include "tools/converter/acl/mapper/stack_mapper.h"
#include "tools/converter/acl/mapper/primitive_mapper_register.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kNameNum = "num";
}

STATUS StackMapper::Mapper(const CNodePtr &cnode) {
  if (AddAttrForDynInputPrimitive(cnode) != RET_OK) {
    MS_LOG(ERROR) << "Stack mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS StackMapper::AddAttrForDynInputPrimitive(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_ASSERT(value_node != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Value node is invalid.";
    return lite::RET_ERROR;
  }
  // add attr input num for dynamic input op
  int64_t num = static_cast<int64_t>(cnode->size());
  if (num > 1) {
    prim->AddAttr(kNameNum, MakeValue(num - 1));
  }
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameStack, StackMapper)
}  // namespace lite
}  // namespace mindspore

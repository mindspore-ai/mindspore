/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/common/gllo_utils.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "ir/anf.h"
#include "base/base_ref.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
bool IsParamNode(const BaseRef &n) {
  if (!utils::isa<ParameterPtr>(n)) {
    return false;
  }
  auto parameter = utils::cast<ParameterPtr>(n);
  if (!parameter->has_default() || parameter->default_param() == nullptr) {
    return false;
  }
  auto tensor = parameter->default_param()->cast<tensor::TensorPtr>();
  if (tensor == nullptr) {
    return false;
  }
  return tensor->data_c() != nullptr;
}

bool CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type) {
  if (node == nullptr || primitive_type == nullptr) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_NULL_PTR);
    return false;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    return IsPrimitive(cnode->input(kAnfPrimitiveIndex), primitive_type);
  } else if (node->isa<ValueNode>()) {
    return IsPrimitive(node, primitive_type);
  }
  return false;
}

}  // namespace opt
}  // namespace mindspore

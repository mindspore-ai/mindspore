/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/unused_cast_node_remove_pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/lite/include/errorcode.h"

namespace mindspore::opt {
constexpr size_t kCastInputNum = 3;
void RemoveUnusedCastOpPass::SetFmkType(FmkType type) { this->fmk_type = type; }

bool RemoveUnusedCastOpPass::Run(const FuncGraphPtr &func_graph) {
  if (this->fmk_type != lite::converter::FmkType_MS) {
    MS_LOG(ERROR) << "The framework type of model should be mindspore.";
    return RET_ERROR;
  }
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (!CheckPrimitiveType(node, prim::kPrimCast)) {
      continue;
    }
    auto cast_cnode = node->cast<CNodePtr>();
    auto abstract_base = cast_cnode->input(1)->abstract();
    if (abstract_base == nullptr) {
      MS_LOG(ERROR) << "Abstract of parameter is nullptr, " << cast_cnode->input(1)->fullname_with_scope();
      return RET_ERROR;
    }
    if (!utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
      MS_LOG(ERROR) << "Abstract of parameter should be abstract tensor, "
                    << cast_cnode->input(1)->fullname_with_scope();
      return RET_ERROR;
    }
    auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
    auto input_type = abstract_tensor->element()->GetTypeTrack();
    MS_ASSERT(input_type != nullptr);
    auto input_type_value = input_type->type_id();

    if (cast_cnode->inputs().size() != kCastInputNum || !utils::isa<ValueNodePtr>(cast_cnode->input(2))) {
      MS_LOG(ERROR) << "Second input of cast should be a ValueNode";
      return RET_ERROR;
    }
    auto output_type = GetValueNode<NumberPtr>(cast_cnode->input(2));
    if (output_type == nullptr) {
      MS_LOG(ERROR) << "Second input of cast is nullptr";
      return RET_ERROR;
    }
    auto output_type_value = output_type->type_id();
    if ((input_type_value == kNumberTypeFloat32 && output_type_value == kNumberTypeFloat16) ||
        (input_type_value == kNumberTypeFloat16 && output_type_value == kNumberTypeFloat32)) {
      manager->Replace(node, cast_cnode->input(1));
    }
  }
  return true;
}
}  // namespace mindspore::opt

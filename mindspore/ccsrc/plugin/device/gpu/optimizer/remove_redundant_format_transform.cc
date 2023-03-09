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

#include "plugin/device/gpu/optimizer/remove_redundant_format_transform.h"
#include <memory>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
const BaseRef RemoveRedundantFormatTransform::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(X);
  MS_EXCEPTION_IF_NULL(Y);
  VectorRef transpose = VectorRef({prim::kPrimTranspose, X, Y});
  return transpose;
}

const AnfNodePtr RemoveRedundantFormatTransform::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                         const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Process node:" << node->fullname_with_scope();
  auto input_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(input_node);
  AnfNodePtr first_transpose = nullptr;
  auto used_node_list = GetRealNodeUsedList(graph, input_node);
  MS_EXCEPTION_IF_NULL(used_node_list);
  for (size_t j = 0; j < used_node_list->size(); j++) {
    auto used_node = used_node_list->at(j).first;
    if (common::AnfAlgo::GetCNodeName(used_node) == prim::kPrimTranspose->name()) {
      first_transpose = used_node;
      break;
    }
  }
  const int64_t perm_param_idx = 1;
  auto first_transpose_perm = AnfAlgo::GetInputDeviceShape(first_transpose, perm_param_idx);
  auto node_perm = AnfAlgo::GetInputDeviceShape(node, perm_param_idx);
  if ((first_transpose != node) && (first_transpose_perm == node_perm)) {
    return first_transpose;
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore

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

#include "backend/optimizer/gpu/remove_format_transform_pair.h"
#include <memory>
#include <vector>
#include <string>
#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef RemoveFormatTransformPair::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(X);
  VectorRef transpose1 = VectorRef({prim::kPrimTranspose, X});
  VectorRef transpose2 = VectorRef({prim::kPrimTranspose, transpose1});
  return transpose2;
}

const AnfNodePtr RemoveFormatTransformPair::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Process node:" << node->fullname_with_scope();
  auto input_node = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(input_node);
  if (AnfAlgo::GetCNodeName(node) != prim::kPrimTranspose->name() ||
      AnfAlgo::GetCNodeName(input_node) != prim::kPrimTranspose->name()) {
    MS_LOG(EXCEPTION) << "The  pattern is not transpose pair, "
                      << "node:" << AnfAlgo::GetCNodeName(node) << " node input:" << AnfAlgo::GetCNodeName(input_node);
  }
  // If transpose operator used by more than one other operators, it cant not be deleted directly.
  if (IsUsedByOthers(graph, input_node)) {
    MS_LOG(DEBUG) << "The transpose node [" << input_node->fullname_with_scope()
                  << "] is used by more than one other operators.";
    return nullptr;
  }
  auto transpose1_input_shape = AnfAlgo::GetInputDeviceShape(input_node, 0);
  auto transpose2_output_shape = AnfAlgo::GetOutputDeviceShape(node, 0);
  if (transpose2_output_shape == transpose1_input_shape) {
    auto transpose1_input_node = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(input_node), 0);
    MS_EXCEPTION_IF_NULL(transpose1_input_node);
    return transpose1_input_node;
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore

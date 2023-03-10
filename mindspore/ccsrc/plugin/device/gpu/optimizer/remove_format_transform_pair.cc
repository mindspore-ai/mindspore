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

#include "plugin/device/gpu/optimizer/remove_format_transform_pair.h"
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
const BaseRef RemoveFormatTransformPair::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  VarPtr Z = std::make_shared<Var>();
  MS_EXCEPTION_IF_NULL(X);
  MS_EXCEPTION_IF_NULL(Y);
  MS_EXCEPTION_IF_NULL(Z);
  VectorRef transpose1 = VectorRef({prim::kPrimTranspose, X, Y});
  VectorRef transpose2 = VectorRef({prim::kPrimTranspose, transpose1, Z});
  return transpose2;
}

const AnfNodePtr RemoveFormatTransformPair::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Process node:" << node->fullname_with_scope();
  auto input_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(input_node);
  if (common::AnfAlgo::GetCNodeName(node) != prim::kPrimTranspose->name() ||
      common::AnfAlgo::GetCNodeName(input_node) != prim::kPrimTranspose->name()) {
    MS_LOG(EXCEPTION) << "The  pattern is not transpose pair, "
                      << "node:" << common::AnfAlgo::GetCNodeName(node)
                      << " node input:" << common::AnfAlgo::GetCNodeName(input_node);
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
    auto transpose1_input_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(input_node), 0);
    MS_EXCEPTION_IF_NULL(transpose1_input_node);
    return transpose1_input_node;
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore

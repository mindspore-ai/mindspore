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

#include "tools/optimizer/fusion/reshape_reshape_fusion.h"
#include "ops/op_utils.h"
#include "ops/reshape.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
namespace {
const auto &p1 = std::placeholders::_1;
}  // namespace

const BaseRef ReshapeReshapeFusion::DefinePattern() const {
  auto reshape1 = VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), reshape_input_,
                             std::make_shared<CondVar>(IsParamNode)});
  return VectorRef({std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape)), reshape1, reshape_shape_});
}

const AnfNodePtr ReshapeReshapeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  auto reshape_prim = std::make_shared<ops::Reshape>();
  if (reshape_prim == nullptr) {
    MS_LOG(ERROR) << "Build reshape primitive failed.";
    return nullptr;
  }
  auto value_node = NewValueNode(reshape_prim);
  auto input = utils::cast<AnfNodePtr>((*equiv)[reshape_input_]);
  auto shape = utils::cast<AnfNodePtr>((*equiv)[reshape_shape_]);
  if (input == nullptr || shape == nullptr) {
    MS_LOG(ERROR) << "Cannot find reshape input and weight.";
    return nullptr;
  }
  // create scale op
  auto new_reshape = func_graph->NewCNode({value_node, input, shape});
  if (new_reshape == nullptr) {
    MS_LOG(ERROR) << "Create new reshape cnode failed.";
    return nullptr;
  }
  return new_reshape;
}
}  // namespace mindspore::opt

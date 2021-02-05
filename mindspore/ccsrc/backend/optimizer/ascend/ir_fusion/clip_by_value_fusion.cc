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
#include "backend/optimizer/ascend/ir_fusion/clip_by_value_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
bool GetMinimumOp(const AnfNodePtr &input0, const AnfNodePtr &input1, CNodePtr *minimum, bool *is_first_input) {
  MS_EXCEPTION_IF_NULL(input0);
  MS_EXCEPTION_IF_NULL(input1);

  CNodePtr cnode = nullptr;
  if (input0->isa<CNode>() && !input1->isa<CNode>()) {
    cnode = input0->cast<CNodePtr>();
    *is_first_input = true;
  } else if (!input0->isa<CNode>() && input1->isa<CNode>()) {
    cnode = input1->cast<CNodePtr>();
    *is_first_input = false;
  } else if (input0->isa<CNode>() && input1->isa<CNode>()) {
    if (AnfAlgo::GetCNodeName(input0) == prim::kPrimMinimum->name()) {
      cnode = input0->cast<CNodePtr>();
      *is_first_input = true;
    } else {
      cnode = input1->cast<CNodePtr>();
      *is_first_input = false;
    }
  } else {
    return false;
  }

  if (AnfAlgo::GetCNodeName(cnode) != prim::kPrimMinimum->name()) {
    return false;
  }
  *minimum = cnode;
  return true;
}
}  // namespace

const BaseRef ClipByValueFusion::DefinePattern() const {
  VectorRef pattern({prim::kPrimMaximum, maximum_input0_, maximum_input1_});
  return pattern;
}

const AnfNodePtr ClipByValueFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                            const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto maximum_input0 = utils::cast<AnfNodePtr>((*equiv)[maximum_input0_]);
  auto maximum_input1 = utils::cast<AnfNodePtr>((*equiv)[maximum_input1_]);
  MS_EXCEPTION_IF_NULL(maximum_input0);
  MS_EXCEPTION_IF_NULL(maximum_input1);

  CNodePtr minimum = nullptr;
  bool is_first_input = true;
  if (!GetMinimumOp(maximum_input0, maximum_input1, &minimum, &is_first_input)) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(minimum);
  if (AnfAlgo::GetInputTensorNum(minimum) != kMinimumInputTensorNum) {
    return nullptr;
  }

  auto prim = std::make_shared<Primitive>(kClipByValueOpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), minimum->input(1),
                                    is_first_input ? maximum_input1 : maximum_input0, minimum->input(2)};
  auto clip_by_value = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(clip_by_value);
  auto types = {AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {AnfAlgo::GetOutputInferShape(node, 0)};
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, clip_by_value.get());
  clip_by_value->set_scope(node->scope());
  return clip_by_value;
}
}  // namespace opt
}  // namespace mindspore
